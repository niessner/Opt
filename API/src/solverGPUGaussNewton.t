local S = require("std")
local util = require("util")
local dbg = require("dbg")
local C = util.C
local Timer = util.Timer

local getValidUnknown = util.getValidUnknown

local gpuMath = util.gpuMath

opt.BLOCK_SIZE = 16
local BLOCK_SIZE =  opt.BLOCK_SIZE

local FLOAT_EPSILON = `0.000001f 
-- GAUSS NEWTON (non-block version)
return function(problemSpec)

	local unknownElement = problemSpec:UnknownType().metamethods.typ
	local unknownType = problemSpec:UnknownType()
    local isGraph = #problemSpec.functions.cost.graphfunctions > 0
    local struct PlanData(S.Object) {
		plan : opt.Plan
		parameters : problemSpec:ParameterType(false)	--get the non-blocked version
		scratchF : &float
		debugDumpImage : &float
		debugCostImage : &float
		debugJTJImage : &float
		debugJTFImage : &float
		debugPreImage : &float


		delta : problemSpec:UnknownType()	--current linear update to be computed -> num vars
		r : problemSpec:UnknownType()		--residuals -> num vars	--TODO this needs to be a 'residual type'
		z : problemSpec:UnknownType()		--preconditioned residuals -> num vars	--TODO this needs to be a 'residual type'
		p : problemSpec:UnknownType()		--descent direction -> num vars
		Ap_X : problemSpec:UnknownType()	--cache values for next kernel call after A = J^T x J x p -> num vars
		preconditioner : problemSpec:UnknownType() --pre-conditioner for linear system -> num vars
		--rDotzOld : problemSpec:UnknownType()	--Old nominator (denominator) of alpha (beta) -> num vars	
		rDotzOld : problemSpec:InternalImage(float, unknownType.metamethods.W, unknownType.metamethods.H, false)	--Old nominator (denominator) of alpha (beta) -> num vars	
		
		scanAlpha : &float					-- tmp variable for alpha scan
		scanBeta : &float					-- tmp variable for alpha scan
		
		timer : Timer
		nIter : int				--current non-linear iter counter
		nIterations : int		--non-linear iterations
		lIterations : int		--linear iterations
	}
	
	
	
	local terra isBlockOnBoundary(gi : int, gj : int, width : uint, height : uint) : bool
		-- TODO meta program the bad based on the block size and stencil;
		-- TODO assert padX > stencil
		
		
		var padX : int
		var padY : int
		escape
			local blockSize = problemSpec:BlockSize()
		
			emit quote
				padX = int(blockSize)
				padY = int(blockSize)
			end
		end
		
		if gi - padX < 0 or gi + padX >= width or 
		   gj - padY < 0 or gj + padY >= height then
			return true
		end
		
		return false
	end
	

    local kernels = {}
    terra kernels.PCGInit1(pd : PlanData)
        var d = 0.0f -- init for out of bounds lanes
        var w : int, h : int
        if getValidUnknown(pd, &w, &h) then
            -- residuum = J^T x -F - A x delta_0  => J^T x -F, since A x x_0 == 0
                            
            var residuum : unknownElement = 0.0f
            var pre : unknownElement = 0.0f	
            if (not [problemSpec:EvalExclude(w,h,w,h,`pd.parameters)]) then 
                            
                residuum, pre = problemSpec.functions.evalJTF.unknownfunction(w, h, w, h, pd.parameters)
                residuum = -residuum
                pd.r(w, h) = residuum
				
				if not problemSpec.usepreconditioner then
					pre = 1.0f
				end
            end        
            	
			if not isGraph then
				if pre(0) > FLOAT_EPSILON then
					pre = 1.0 / pre
				else 
					pre = 1.0
				end
							
				var p = pre*residuum	-- apply pre-conditioner M^-1			   
				pd.p(w, h) = p
				
				--d = residuum*p		-- x-th term of nominator for computing alpha and denominator for computing beta
				d = util.Dot(residuum,p) 
			end
			
			pd.preconditioner(w, h) = pre
			
        end 
		if not isGraph then
			d = util.warpReduce(d)	
			if (util.laneid() == 0) then
				util.atomicAdd(pd.scanAlpha, d)
			end
		end
    end
    
    terra kernels.PCGInit1_Graph(pd : PlanData)
		var tIdx = 0 	
		escape 
	    	for i,func in ipairs(problemSpec.functions.evalJTF.graphfunctions) do
				local name,implementation = func.graphname,func.implementation
				emit quote 
	    			if util.getValidGraphElement(pd,[name],&tIdx) then
						implementation(tIdx, pd.parameters, pd.r, pd.preconditioner)
	    			end 
				end
    		end
    	end
    end

    terra kernels.PCGInit1_Finish(pd : PlanData)	--only called for graphs
    	var d = 0.0f -- init for out of bounds lanes
        var w : int, h : int
        if getValidUnknown(pd, &w, &h) then
        	var residuum = pd.r(w, h)			
			var pre = pd.preconditioner(w, h)
			
			if pre(0) > FLOAT_EPSILON then
				pre = 1.0 / pre
			else 
				pre = 1.0
			end
			
			if not problemSpec.usepreconditioner then
				pre = 1.0f
			end
			
			var p = pre*residuum	-- apply pre-conditioner M^-1			   
			pd.p(w, h) = p
        	d = util.Dot(residuum, p)
        end

		d = util.warpReduce(d)	
        if (util.laneid() == 0) then
            util.atomicAdd(pd.scanAlpha, d)
        end
	end
	
    terra kernels.PCGInit2(pd : PlanData)
        var w : int, h : int
        if getValidUnknown(pd, &w, &h) and (not [problemSpec:EvalExclude(w,h,w,h,`pd.parameters)]) then
            pd.rDotzOld(w,h) = pd.scanAlpha[0]
            pd.delta(w,h) = 0.0f	--TODO check if we need that
        end
    end
	
    terra kernels.PCGStep1(pd : PlanData)
        var d = 0.0f
        var w : int, h : int
        if getValidUnknown(pd, &w, &h) and (not [problemSpec:EvalExclude(w,h,w,h,`pd.parameters)]) then
            var tmp : unknownElement = 0.0f
             -- A x p_k  => J^T x J x p_k 
            tmp = problemSpec.functions.applyJTJ.unknownfunction(w, h, w, h, pd.parameters, pd.p)
            pd.Ap_X(w, h) = tmp								  -- store for next kernel call
            --d = pd.p(w, h)*tmp					              -- x-th term of denominator of alpha
            d = util.Dot(pd.p(w,h),tmp)
        end
        d = util.warpReduce(d)
        if (util.laneid() == 0) then
            util.atomicAdd(pd.scanAlpha, d)
        end
    end
	
	terra kernels.PCGStep1_Graph(pd : PlanData)
		var d = 0.0f
		var tIdx = 0 	
        escape 
			for i,func in ipairs(problemSpec.functions.applyJTJ.graphfunctions) do
				local name,implementation = func.graphname,func.implementation
				emit quote 
				    if util.getValidGraphElement(pd,[name],&tIdx) then
				        d = d + implementation(tIdx, pd.parameters, pd.p, pd.Ap_X)
				    end 
				end
			end
		end
		d = util.warpReduce(d)
        if (util.laneid() == 0) then
            util.atomicAdd(pd.scanAlpha, d)
        end
    end

	
	terra kernels.PCGStep2(pd : PlanData)
        var b = 0.0f 
        var w : int, h : int
        if getValidUnknown(pd, &w, &h)  and (not [problemSpec:EvalExclude(w,h,w,h,`pd.parameters)]) then
            -- sum over block results to compute denominator of alpha
            var dotProduct : float = pd.scanAlpha[0]
            var alpha = 0.0f
            
            -- update step size alpha
            if dotProduct > FLOAT_EPSILON then alpha = pd.rDotzOld(w, h)/dotProduct end 
        
            pd.delta(w, h) = pd.delta(w, h)+alpha*pd.p(w,h)		-- do a decent step
            
            var r = pd.r(w,h)-alpha*pd.Ap_X(w,h)				-- update residuum
            pd.r(w,h) = r										-- store for next kernel call
        
			var pre = pd.preconditioner(w,h)
			if not problemSpec.usepreconditioner then
				pre = 1.0f
			end
			
			if isGraph then
				if pre(0) > FLOAT_EPSILON then
					pre = 1.0 / pre
				else 
					pre = 1.0
				end
			end
			
            var z = pre*r										-- apply pre-conditioner M^-1
            pd.z(w,h) = z;										-- save for next kernel call
            
            --b = z*r;											-- compute x-th term of the nominator of beta
            b = util.Dot(z,r)
        end
        b = util.warpReduce(b)
        if (util.laneid() == 0) then
            util.atomicAdd(pd.scanBeta, b)
        end
    end
	
    terra kernels.PCGStep3(pd : PlanData)			
        var w : int, h : int
        if getValidUnknown(pd, &w, &h) and (not [problemSpec:EvalExclude(w,h,w,h,`pd.parameters)]) then
            var rDotzNew : float =  pd.scanBeta[0]						-- get new nominator
            var rDotzOld : float = pd.rDotzOld(w,h)						-- get old denominator

            var beta : float = 0.0f			
                    
            if rDotzOld > FLOAT_EPSILON then beta = rDotzNew/rDotzOld end	-- update step size beta
        
            pd.rDotzOld(w,h) = rDotzNew										-- save new rDotz for next iteration
            pd.p(w,h) = pd.z(w,h)+beta*pd.p(w,h)							-- update decent direction
        end
    end
	
	terra kernels.PCGLinearUpdate(pd : PlanData)
        var w : int, h : int
        if getValidUnknown(pd, &w, &h) and (not [problemSpec:EvalExclude(w,h,w,h,`pd.parameters)]) then
            pd.parameters.X(w,h) = pd.parameters.X(w,h) + pd.delta(w,h)
        end
    end	
	
	terra kernels.computeCost(pd : PlanData)			
        var cost : float = 0.0f
        var w : int, h : int
        if util.getValidUnknown(pd, &w, &h)  and (not [problemSpec:EvalExclude(w,h,w,h,`pd.parameters)]) then
            var params = pd.parameters				
            cost = cost + [float](problemSpec.functions.cost.unknownfunction(w, h, w, h, params))
        end

        cost = util.warpReduce(cost)
        if (util.laneid() == 0) then
        util.atomicAdd(pd.scratchF, cost)
        end
    end
	
	terra kernels.computeCost_Graph(pd : PlanData)			
        var cost : float = 0.0f
		
		var tIdx = 0 	
        escape 
			for i,func in ipairs(problemSpec.functions.cost.graphfunctions) do
				local name,implementation = func.graphname,func.implementation
				emit quote 
				    if util.getValidGraphElement(pd,[name],&tIdx) then
				        cost = cost + implementation(tIdx, pd.parameters)
				    end 
				end
			end
		end
		
        cost = util.warpReduce(cost)
        if (util.laneid() == 0) then
            util.atomicAdd(pd.scratchF, cost)
        end
    end
    
    terra kernels.precomputeImages(pd : PlanData)
        var w : int, h : int
        if util.getValidUnknown(pd,&w,&h) then
            escape
                if problemSpec.functions.precompute then
                    emit quote 
                        problemSpec.functions.precompute.unknownfunction(w,h,w,h,pd.parameters)
                    end
                end
            end
        end
    end

	if util.debugDumpInfo then
	    terra kernels.dumpCostJTFAndPre(pd : PlanData)
			var w : int, h : int
			if getValidUnknown(pd, &w, &h) then
				-- residuum = J^T x -F - A x delta_0  => J^T x -F, since A x x_0 == 0
				
				var residuum : unknownElement = 0.0f
				var pre : unknownElement = 0.0f
				var cost : float = 0.0f

				cost = problemSpec.functions.cost.unknownfunction(w, h, w, h, pd.parameters)
				residuum, pre = problemSpec.functions.evalJTF.unknownfunction(w, h, w, h, pd.parameters)

				pd.debugCostImage[h*pd.parameters.X:W()+w] = cost
				[&unknownElement](pd.debugJTFImage)[h*pd.parameters.X:W()+w]  = residuum
				[&unknownElement](pd.debugPreImage)[h*pd.parameters.X:W()+w]  = pre
			end 
		end
	end

	terra kernels.dumpJTJ(pd : PlanData)
        var d = 0.0f
        var w : int, h : int
        if getValidUnknown(pd, &w, &h) then
            var tmp : unknownElement = 0.0f
             -- A x p_k  => J^T x J x p_k 
            tmp = problemSpec.functions.applyJTJ.unknownfunction(w, h, w, h, pd.parameters, pd.p)
            [&unknownElement](pd.debugJTJImage)[h*pd.parameters.X:W()+w] = tmp
        end
    end
    
	local gpu = util.makeGPUFunctions(problemSpec, PlanData, kernels)

	---------------------------------------DEBUGGING FUNCTIONS------------------------------------------

	local terra initAllDebugImages(pd : &PlanData)
		C.printf("initAllDebugImages\n")
		dbg.initCudaImage(&pd.debugDumpImage,pd.parameters.X:W(), pd.parameters.X:H(),  1)
		dbg.initCudaImage(&pd.debugCostImage,pd.parameters.X:W(), pd.parameters.X:H(),  1)
		dbg.initCudaImage(&pd.debugJTJImage, pd.parameters.X:W(), pd.parameters.X:H(), sizeof([unknownElement]) / 4)
		dbg.initCudaImage(&pd.debugJTFImage, pd.parameters.X:W(), pd.parameters.X:H(), sizeof([unknownElement]) / 4)
		dbg.initCudaImage(&pd.debugPreImage, pd.parameters.X:W(), pd.parameters.X:H(), sizeof([unknownElement]) / 4)
	end




	local terra dumpImage(pd : &PlanData, ptr: &float, name : rawstring, nIter : int, lIter : int)
		if ([util.debugDumpInfo]) then
			var buffer : int8[64]
			C.sprintf(buffer, "%s_%d_%d.imagedump", name, nIter, lIter)
			dbg.imageWriteFromCuda(pd, ptr, 1, buffer)
		end
	end

	local function tableLength(T)
	   local count = 0
	   for _ in pairs(T) do count = count + 1 end
	   return count
	end


	---------------------------------------END DEBUGGING FUNCTIONS------------------------------------------

	
	local terra init(data_ : &opaque, images : &&opaque, graphSizes : &int32, edgeValues : &&opaque, xs : &&int32, ys : &&int32, params_ : &&opaque, solverparams : &&opaque)
	   var pd = [&PlanData](data_)
	   pd.timer:init()
       [util.initParameters(`pd.parameters,problemSpec,images, graphSizes,edgeValues,xs,ys,params_,true)]
	   pd.nIter = 0
	   
	   escape 
	      if util.debugDumpInfo then
		 emit quote
		    C.printf("initDebugImages\n")
		    initAllDebugImages(pd)
		 end
	      end
	   end
	   var buffer : int8[64]
	   escape
	      --TODO: Remove
 	     if false then
		 for key, entry in ipairs(problemSpec.parameters) do
		    if entry.kind == "image" then
		       print("" .. key .. " = imageTypes[" .. entry.idx .. "] = " .. tostring(entry.type))
		       local channel = 0 
		       local pixelType = entry.type.metamethods.typ
		       if pixelType == float then
			  channel = 1 
		       elseif pixelType == opt.float2 then
			  channel = 2
		       elseif pixelType == opt.float3 then
			  channel = 3
		       elseif pixelType == opt.float4 then
			  channel = 4
		       end
		       emit quote
			     C.sprintf(buffer, "arapTest%d.imagedump", entry.idx)
			     dbg.imageWriteFromCuda(pd, images[entry.idx], channel, buffer)
		       end

		    end
		 end

	      
	      end
	   end
           pd.nIterations = @[&int](solverparams[0])
	   pd.lIterations = @[&int](solverparams[1])
	end
    local terra computeCost(pd : &PlanData) : float
        C.cudaMemset(pd.scratchF, 0, sizeof(float))
        gpu.computeCost(pd)
        gpu.computeCost_Graph(pd)
        var f : float
        C.cudaMemcpy(&f, pd.scratchF, sizeof(float), C.cudaMemcpyDeviceToHost)
        return f
    end

	local terra step(data_ : &opaque, images : &&opaque, graphSizes : &int32, edgeValues : &&opaque, xs : &&int32, ys : &&int32, params_ : &&opaque, solverparams : &&opaque)
		var pd = [&PlanData](data_)
		[util.initParameters(`pd.parameters,problemSpec, images, graphSizes,edgeValues,xs,ys,params_,false)]
        escape 
	    	if util.debugDumpInfo then
	    		emit quote

	    			-- Hack for SFS
	    			--var solveCount = ([&&uint32](params_))[39][0]
	    			if pd.nIter == 0 then
		    			C.printf("dumpingCostJTFAndPre\n")
		    			gpu.dumpCostJTFAndPre(pd)
		    			C.printf("saving\n")
		    			dbg.imageWriteFromCudaPrefix(pd.debugCostImage, pd.parameters.X:W(), pd.parameters.X:H(), 1, "cost")
		    			dbg.imageWriteFromCudaPrefix(pd.debugJTFImage, pd.parameters.X:W(), pd.parameters.X:H(), sizeof([unknownElement]) / 4, "JTF")
		    			dbg.imageWriteFromCudaPrefix(pd.debugPreImage, pd.parameters.X:W(), pd.parameters.X:H(), sizeof([unknownElement]) / 4, "Pre")
		    		end


	    		end
	    	end
		end
		
        gpu.precomputeImages(pd)    
		if pd.nIter < pd.nIterations then
		    var startCost = computeCost(pd)
			logSolver("iteration %d, cost=%f\n", pd.nIter, startCost)

			C.cudaMemset(pd.scanAlpha, 0, sizeof(float))	--scan in PCGInit1 requires reset
			gpu.PCGInit1(pd)
			if isGraph then
				C.cudaMemset(pd.scanAlpha, 0, sizeof(float))	--TODO: don't write to scanAlpha in the previous kernel if it is a graph
				gpu.PCGInit1_Graph(pd)	
				gpu.PCGInit1_Finish(pd)	
			end
			gpu.PCGInit2(pd)
			
			escape
			    if util.debugPrintSolverInfo then
				emit quote
				    var temp : float
				    C.cudaMemcpy(&temp, pd.scanAlpha, sizeof(float), C.cudaMemcpyDeviceToHost)
				    C.printf("ScanAlpha (Init): %f\n", temp);


				    --[[
				    var r0 : float
				    var r1 : float
				    var r2 : float
				    var r3 : float
				    C.cudaMemcpy(&r0, ([&float](pd.r.data)) + 0, sizeof(float), C.cudaMemcpyDeviceToHost)
				    C.cudaMemcpy(&r1, ([&float](pd.r.data)) + 1, sizeof(float), C.cudaMemcpyDeviceToHost)
				    C.cudaMemcpy(&r2, ([&float](pd.r.data)) + 2, sizeof(float), C.cudaMemcpyDeviceToHost)
				    C.cudaMemcpy(&r3, ([&float](pd.r.data)) + 3, sizeof(float), C.cudaMemcpyDeviceToHost)
				    C.printf("Residuum 0: %f, %f, %f, %f\n", r0, r1, r2, r3);

				    C.cudaMemcpy(&r0, ([&float](pd.r.data)) + 4, sizeof(float), C.cudaMemcpyDeviceToHost)
				    C.cudaMemcpy(&r1, ([&float](pd.r.data)) + 5, sizeof(float), C.cudaMemcpyDeviceToHost)
				    C.cudaMemcpy(&r2, ([&float](pd.r.data)) + 6, sizeof(float), C.cudaMemcpyDeviceToHost)
				    C.cudaMemcpy(&r3, ([&float](pd.r.data)) + 7, sizeof(float), C.cudaMemcpyDeviceToHost)
				    C.printf("Residuum 1: %f, %f, %f, %f\n", r0, r1, r2, r3);


				    C.cudaMemcpy(&r0, ([&float](pd.preconditioner.data)) + 0, sizeof(float), C.cudaMemcpyDeviceToHost)
				    C.cudaMemcpy(&r1, ([&float](pd.preconditioner.data)) + 1, sizeof(float), C.cudaMemcpyDeviceToHost)
				    C.cudaMemcpy(&r2, ([&float](pd.preconditioner.data)) + 2, sizeof(float), C.cudaMemcpyDeviceToHost)
				    C.cudaMemcpy(&r3, ([&float](pd.preconditioner.data)) + 3, sizeof(float), C.cudaMemcpyDeviceToHost)
				    C.printf("Preconditioner 0: %f, %f, %f, %f\n", r0, r1, r2, r3);

				    C.cudaMemcpy(&r0, ([&float](pd.preconditioner.data)) + 4, sizeof(float), C.cudaMemcpyDeviceToHost)
				    C.cudaMemcpy(&r1, ([&float](pd.preconditioner.data)) + 5, sizeof(float), C.cudaMemcpyDeviceToHost)
				    C.cudaMemcpy(&r2, ([&float](pd.preconditioner.data)) + 6, sizeof(float), C.cudaMemcpyDeviceToHost)
				    C.cudaMemcpy(&r3, ([&float](pd.preconditioner.data)) + 7, sizeof(float), C.cudaMemcpyDeviceToHost)
				    C.printf("Preconditioner 1: %f, %f, %f, %f\n", r0, r1, r2, r3);
--]]
				end
			    end

			    if util.debugDumpInfo then

				    emit quote
			    		-- Hack for SFS
		    			--var solveCount = ([&&uint32](params_))[39][0]
		    			
		    			if pd.nIter == 0 then
			    			C.printf("dumpingJTJ\n")
			    			gpu.dumpJTJ(pd)
			    			C.printf("saving\n")
			    			dbg.imageWriteFromCudaPrefix(pd.debugJTJImage, pd.parameters.X:W(), pd.parameters.X:H(), sizeof([unknownElement]) / 4, "JTJ")
			    			return 0
			    		end
		    		end
		    	end
		    end

			for lIter = 0, pd.lIterations do
				

                C.cudaMemset(pd.scanAlpha, 0, sizeof(float))
				gpu.PCGStep1(pd)

				if isGraph then
					gpu.PCGStep1_Graph(pd)	
				end
				
				C.cudaMemset(pd.scanBeta, 0, sizeof(float))
				gpu.PCGStep2(pd)
				gpu.PCGStep3(pd)


				escape
				    if util.debugPrintSolverInfo then
					emit quote
					var temp : float
					C.cudaMemcpy(&temp, pd.scanAlpha, sizeof(float), C.cudaMemcpyDeviceToHost)
					C.printf("ScanAlpha (Step): %f\n", temp);
					C.cudaMemcpy(&temp, pd.scanBeta, sizeof(float), C.cudaMemcpyDeviceToHost)
					C.printf("ScanBeta (Step): %f\n", temp);		
					end
				    end
			        end

			end
			
			gpu.PCGLinearUpdate(pd)
		    pd.nIter = pd.nIter + 1
		    return 1
		else
			escape
				if util.debugDumpInfo then
		    		emit quote
						dbg.imageWriteFromCudaPrefix([&float](pd.parameters.X.data), pd.parameters.X:W(), pd.parameters.X:H(), sizeof([unknownElement]) / 4, "result")

					end
				end
			end
			var finalCost = computeCost(pd)
			logSolver("final cost=%f\n", finalCost)
		    pd.timer:evaluate()
		    pd.timer:cleanup()
		    return 0
		end
	end

	local terra makePlan() : &opt.Plan
		var pd = PlanData.alloc()
		pd.plan.data = pd
		pd.plan.init,pd.plan.step = init,step

		pd.delta:initGPU()
		pd.r:initGPU()
		pd.z:initGPU()
		pd.p:initGPU()
		pd.Ap_X:initGPU()
		pd.preconditioner:initGPU()
		pd.rDotzOld:initGPU()
		
		C.cudaMalloc([&&opaque](&(pd.scanAlpha)), sizeof(float))
		C.cudaMalloc([&&opaque](&(pd.scanBeta)), sizeof(float))
		
		C.cudaMalloc([&&opaque](&(pd.scratchF)), sizeof(float))
		return &pd.plan
	end
	return makePlan
end
