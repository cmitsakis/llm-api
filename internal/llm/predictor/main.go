package predictor

import (
	"fmt"

	llama "github.com/go-skynet/go-llama.cpp"
)

type Predictor struct {
	llm               *llama.LLama
	predictOptionArgs []llama.PredictOption
}

func New(modelPath string, modelOptionArgs []llama.ModelOption, predictOptionArgs []llama.PredictOption) (Predictor, error) {
	l, err := llama.New(modelPath, modelOptionArgs...)
	if err != nil {
		return Predictor{}, fmt.Errorf("Loading the model failed: %w", err)
	}
	return Predictor{
		llm:               l,
		predictOptionArgs: predictOptionArgs,
	}, nil
}

func (p Predictor) Predict(prompt string, predictOptionArgs ...llama.PredictOption) (string, error) {
	var opts []llama.PredictOption
	if len(predictOptionArgs) == 0 {
		opts = p.predictOptionArgs
	} else {
		opts = make([]llama.PredictOption, len(p.predictOptionArgs))
		copy(opts, p.predictOptionArgs)
		opts = append(opts, predictOptionArgs...)
	}
	response, err := p.llm.Predict(prompt, opts...)
	if err != nil {
		return "", fmt.Errorf("Predict() failed: %w", err)
	}
	return response, nil
}

func (p Predictor) PredictToChannel(prompt string, responseChan chan<- string, predictOptionArgs ...llama.PredictOption) (string, error) {
	defer close(responseChan)
	predictOptionArgs = append(predictOptionArgs, llama.SetTokenCallback(func(token string) bool {
		responseChan <- token
		return true
	}))
	return p.Predict(prompt, predictOptionArgs...)
}

func (p Predictor) Free() {
	p.llm.Free()
}
