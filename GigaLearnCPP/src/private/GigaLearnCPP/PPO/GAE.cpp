#include "GAE.h"

void GGL::GAE::Compute(
	torch::Tensor rews, torch::Tensor terminals, torch::Tensor valPreds, torch::Tensor truncValPreds,
	torch::Tensor& outAdvantages, torch::Tensor& outTargetValues, torch::Tensor& outReturns, float& outRewClipPortion,
	float gamma, float lambda, float returnStd, float clipRange
) {

	bool hasTruncValPreds = truncValPreds.defined();

	if (!(gamma >= 0.0f && gamma <= 1.0f) || !std::isfinite(gamma))
		RG_ERR_CLOSE("GAE: gamma must be finite and in [0, 1]");
	if (!(lambda >= 0.0f && lambda <= 1.0f) || !std::isfinite(lambda))
		RG_ERR_CLOSE("GAE: lambda must be finite and in [0, 1]");
	if (returnStd < 0.0f || !std::isfinite(returnStd))
		RG_ERR_CLOSE("GAE: returnStd must be finite and >= 0");
	if (clipRange < 0.0f || !std::isfinite(clipRange))
		RG_ERR_CLOSE("GAE: clipRange must be finite and >= 0");

	if (!rews.device().is_cpu() || !terminals.device().is_cpu() || !valPreds.device().is_cpu() ||
		(hasTruncValPreds && !truncValPreds.device().is_cpu()))
		RG_ERR_CLOSE("GAE: Compute expects CPU tensors");

	if (rews.scalar_type() != torch::kFloat32 || valPreds.scalar_type() != torch::kFloat32 ||
		(hasTruncValPreds && truncValPreds.scalar_type() != torch::kFloat32) ||
		terminals.scalar_type() != torch::kInt8)
		RG_ERR_CLOSE("GAE: expected rewards/values float32 and terminals int8");

	float prevLambda = 0;
	int numReturns = rews.size(0);
	if (terminals.size(0) != numReturns)
		RG_ERR_CLOSE("GAE: rewards and terminals must have the same number of returns");
	if (valPreds.size(0) != numReturns + 1)
		RG_ERR_CLOSE("GAE: valPreds must contain numReturns + 1 values for bootstrap (got " << valPreds.size(0) << ", expected " << numReturns + 1 << ")");

	outAdvantages = torch::zeros(numReturns);
	outReturns = torch::zeros(numReturns);
	float prevRet = 0;
	int truncCount = 0;

	float totalRew = 0, totalClippedRew = 0;

	rews = rews.contiguous();
	terminals = terminals.contiguous();
	valPreds = valPreds.contiguous();
	if (hasTruncValPreds)
		truncValPreds = truncValPreds.contiguous();

	auto _terminals = terminals.const_data_ptr<int8_t>();
	auto _rews = rews.const_data_ptr<float>();
	auto _valPreds = valPreds.const_data_ptr<float>();

	const float* _truncValPreds;
	int numTruncs;
	if (hasTruncValPreds) {
		_truncValPreds = truncValPreds.const_data_ptr<float>();
		numTruncs = truncValPreds.size(0);
	} else {
		_truncValPreds = NULL;
		numTruncs = 0;
	}

	auto _outReturns = std::vector<float>(numReturns, 0);
	auto _outAdvantages = std::vector<float>(numReturns, 0);

	for (int step = numReturns - 1; step >= 0; step--) {
		uint8_t terminal = _terminals[step];
		float done = terminal == RLGC::TerminalType::NORMAL;
		float trunc = terminal == RLGC::TerminalType::TRUNCATED;

		float curReward;
		if (returnStd != 0) {
			curReward = _rews[step] / returnStd;
			totalRew += abs(curReward);

			if (clipRange > 0) {
				curReward = RS_CLAMP(curReward, -clipRange, clipRange);
				totalClippedRew += abs(curReward);
			}
		} else {
			curReward = _rews[step];
			totalRew += abs(curReward);
		}

		float nextValPred;
		if (terminal == RLGC::TerminalType::TRUNCATED) {
			if (!hasTruncValPreds)
				RG_ERR_CLOSE("GAE encountered a truncated terminal, but has no truncated val pred");

			if (truncCount >= numTruncs)
				RG_ERR_CLOSE("GAE encountered too many truncated terminals, not enough val preds (max: " << numTruncs << ")")

			nextValPred = _truncValPreds[truncCount];
			truncCount++;
		} else {
			nextValPred = _valPreds[step + 1];
		}

		float predReturn = curReward + gamma * nextValPred * (1 - done);
		float delta = predReturn - _valPreds[step];
		float curReturn = _rews[step] + prevRet * gamma * (1 - done) * (1 - trunc);
		_outReturns[step] = curReturn;

		prevLambda = delta + gamma * lambda * (1 - done) * (1 - trunc) * prevLambda;
		_outAdvantages[step] = prevLambda;

		prevRet = curReturn;
	}

	if (hasTruncValPreds)
		if (truncCount != truncValPreds.size(0))
			RG_ERR_CLOSE("GAE didn't receive expected truncation count (only " << truncCount << "/" << truncValPreds.size(0) << ")");

	outReturns = torch::tensor(_outReturns);
	outAdvantages = torch::tensor(_outAdvantages);
	outTargetValues = valPreds.slice(0, 0, numReturns) + outAdvantages;

	if (returnStd == 0 || clipRange <= 0 || totalRew <= 0)
		outRewClipPortion = 0;
	else
		outRewClipPortion = (totalRew - totalClippedRew) / totalRew;
}