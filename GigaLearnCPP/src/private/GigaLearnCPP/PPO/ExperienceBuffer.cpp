#include "ExperienceBuffer.h"

using namespace torch;

GGL::ExperienceBuffer::ExperienceBuffer(int seed, torch::Device device) :
	seed(seed), device(device), rng(seed) {

}

GGL::ExperienceTensors GGL::ExperienceBuffer::_GetSamples(const int64_t* indices, size_t size) const {

	// TODO: Slow, use blob
	// index_select requires indices to live on the same device as the source tensor.
	// ExperienceBuffer is normally CPU-resident, but keeping this device-aware
	// avoids a silent CPU/CUDA mismatch if the buffer is ever moved.
	auto indexOptions = torch::TensorOptions()
		.dtype(torch::kInt64)
		.device(data.states.device());
	Tensor tIndices = torch::tensor(IList(indices, indices + size), indexOptions);

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
	if (expSize == 0)
		return {};

	int64_t* indices = new int64_t[expSize];
	std::iota(indices, indices + expSize, 0);
	std::shuffle(indices, indices + expSize, rng);

	std::vector<ExperienceTensors> result;
	// Retain the final undersized batch so no collected experience is silently
	// discarded, including when the entire buffer is smaller than batchSize.
	for (int64_t startIdx = 0; startIdx < (int64_t) expSize; startIdx += batchSize) {
		int64_t curBatchSize = std::min<int64_t>(batchSize, (int64_t) expSize - startIdx);

		if (overbatching && curBatchSize == batchSize && startIdx + batchSize * 2 > (int64_t) expSize)
			curBatchSize = (int64_t) expSize - startIdx;

		result.push_back(_GetSamples(indices + startIdx, (size_t) curBatchSize));
	}

	delete[] indices;
	return result;
}