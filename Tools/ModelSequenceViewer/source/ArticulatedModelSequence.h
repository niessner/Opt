#pragma once
#include <G3D/G3DAll.h>
class ArticulatedModelSequence : public VisibleEntity {
public:
	virtual bool poseModel(Array<shared_ptr<Surface> >& surfaceArray) const override;
	virtual void onSimulation(SimTime absoluteTime, SimTime deltaTime) override;
	void loadFromDirectory(const String& directory);

	static shared_ptr<ArticulatedModelSequence> create(const String& directory, const String& name) {
		auto s = shared_ptr<ArticulatedModelSequence>(new ArticulatedModelSequence());
		s->loadFromDirectory(directory);
		s->m_name = name;
		return s;
	}
protected:
	Array<shared_ptr<ArticulatedModel>> m_models;
	RealTime m_framerate = 60.0f;
	size_t m_currentIndex = 0;
};