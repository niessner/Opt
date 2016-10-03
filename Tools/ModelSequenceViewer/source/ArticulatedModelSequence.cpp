#include "ArticulatedModelSequence.h"


bool ArticulatedModelSequence::poseModel(Array<shared_ptr<Surface> >& surfaceArray) const {
	if (m_currentIndex < m_models.size()) {
		const shared_ptr<Entity>& me = dynamic_pointer_cast<Entity>(const_cast<ArticulatedModelSequence*>(this)->shared_from_this());
		m_models[m_currentIndex]->pose(surfaceArray, m_frame, m_artPose, m_previousFrame, m_artPreviousPose, me);
		return true;
	}
	return false;
}
void ArticulatedModelSequence::onSimulation(SimTime absoluteTime, SimTime deltaTime) {
	m_currentIndex = iClamp(absoluteTime * m_framerate, 0, m_models.size() - 1);
}
void ArticulatedModelSequence::loadFromDirectory(const String& directory) {
	Array<String> filenames;
	FileSystem::getFiles(directory + "/*.*", filenames);
	for (int i = 0; i < filenames.size(); ++i) {
		auto f = filenames[i];
		ArticulatedModel::Specification spec;
		spec.filename = directory + "/" + f;
		spec.stripMaterials = true;
		spec.stripVertexColors = true;
		m_models.append(ArticulatedModel::create(spec));
		debugPrintf("Loaded model: %d\n", i);
	}
}