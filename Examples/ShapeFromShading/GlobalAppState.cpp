#include "stdafx.h"

#include "GlobalAppState.h"


HRESULT GlobalAppState::OnD3D11CreateDevice(ID3D11Device* pd3dDevice)
{
	HRESULT hr = S_OK;

	/////////////////////////////////////////////////////
	// Query
	/////////////////////////////////////////////////////

	D3D11_QUERY_DESC queryDesc;
	queryDesc.Query = D3D11_QUERY_EVENT;
	queryDesc.MiscFlags = 0;

	hr = pd3dDevice->CreateQuery(&queryDesc, &s_pQuery);
	if(FAILED(hr)) return hr;

	return  hr;
}

void GlobalAppState::OnD3D11DestroyDevice()
{
	SAFE_RELEASE(s_pQuery);
}

void GlobalAppState::WaitForGPU()
{
	DXUTGetD3D11DeviceContext()->Flush();
	DXUTGetD3D11DeviceContext()->End(s_pQuery);
	DXUTGetD3D11DeviceContext()->Flush();

	while (S_OK != DXUTGetD3D11DeviceContext()->GetData(s_pQuery, NULL, 0, 0));
}