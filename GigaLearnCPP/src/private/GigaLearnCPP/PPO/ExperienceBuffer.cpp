#include "ExperienceBuffer.h"

using namespace torch;

GGL::ExperienceBuffer::ExperienceBuffer(int seed, torch::Device device) :
	seed(seed), device(device), rng(seed) {

}

GGL::ExperienceTensors GGL::ExperienceBuffer::_GetSamples(const int64_t* indices, size_t size) const {

	// TODO: Slow, use blob
	Tensor tIndices = torch::tensor(IList(indices, indices + size));

	ExperienceTensors result;
	auto* toItr = result.begin();
	auto* fromItr = data.begin();
	for (; toItr != result.end(); toItr++, fromItr++)
		*toItr = torch::index_select(*fromItr, 0, tIndices);

	return result;
}

std::vector<GGL::ExperienceTensors> GGL::ExperienceBuffer::GetAllBatchesShuffled(int64_t batchSize, bool overbatching) {

	RG_NO_GRAD;

	if (batchSize <= 0)
		RG_ERR_CLOSE("ExperienceBuffer: batchSize must be greater than 0");

	size_t expSize = data.states.size(0);

	// Make list of shuffled sample indices
	int64_t* indices = new int64_t[expSize];
	std::iota(indices, indices + expSize, 0); // Fill ascending indices
	std::shuffle(indices, indices + expSize, rng);

	// Get a sample set from each of the batches. When overbatching is enabled,
	// the final batch may contain more than batchSize samples; return the entire
	// tail so callers can decide how to split/process it.
	std::vector<ExperienceTensors> result;
	for (int64_t startIdx = 0; startIdx < (int64_t) expSize; startIdx += batchSize) {

		int64_t remaining = (int64_t) expSize - startIdx;
		int64_t curBatchSize = std::min<int64_t>(batchSize, remaining);

		if (curBatchSize < batchSize && !overbatching)
			break;

		result.push_back(_GetSamples(indices + startIdx, (size_t) curBatchSize));
	}

	delete[] indices;
	return result;
}