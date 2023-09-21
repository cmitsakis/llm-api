package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"text/template"
	"time"

	llama "github.com/go-skynet/go-llama.cpp"

	"cmitsakis/llm-api/internal/llm/conversation"
	"cmitsakis/llm-api/internal/llm/predictor"
)

var predictMutex sync.Mutex

func handlePrediction(w http.ResponseWriter, r *http.Request, p predictor.Predictor, prompt string, stopRegex *regexp.Regexp) {
	log.Printf("<prompt>%s</prompt>\n", prompt)
	stopRegexSubmittedStr := r.Form.Get("stopRegex")
	var stopRegexSubmitted *regexp.Regexp
	if stopRegexSubmittedStr != "" {
		log.Printf("<stopRegex>%s</stopRegex>\n", stopRegexSubmittedStr)
		var err error
		stopRegexSubmitted, err = regexp.Compile(stopRegexSubmittedStr)
		if err != nil {
			w.WriteHeader(http.StatusBadRequest)
			fmt.Fprintf(w, "failed to parse stopRegex: %s", err)
			return
		}
	}
	var tokensAccumulated string
	opts := []llama.PredictOption{llama.SetTokenCallback(func(token string) bool {
		tokensAccumulated, token = conversation.TrimAndAppend(tokensAccumulated, token)
		if stopRegex != nil && stopRegex.MatchString(tokensAccumulated) || stopRegexSubmitted != nil && stopRegexSubmitted.MatchString(tokensAccumulated) {
			return false
		}
		_, err := io.WriteString(w, token)
		if err != nil {
			return false
		}
		return true
	})}
	temperatureStr := r.Form.Get("temperature")
	if temperatureStr != "" {
		temperature, err := strconv.ParseFloat(temperatureStr, 32)
		if err != nil {
			w.WriteHeader(http.StatusBadRequest)
			fmt.Fprintf(w, "failed to parse value 'temperature' %s: %s", temperatureStr, err)
			return
		}
		opts = append(opts, llama.SetTemperature(float32(temperature)))
		log.Printf("<temperature>%v</temperature>\n", temperature)
	}
	locked := predictMutex.TryLock()
	if !locked {
		// another request is performing prediction
		// reject this request with HTTP 503
		log.Printf("sending HTTP error: %v. Server is busy", http.StatusText(http.StatusServiceUnavailable))
		w.WriteHeader(http.StatusServiceUnavailable)
		fmt.Fprintf(w, "server is busy")
		return
	}
	defer predictMutex.Unlock()
	response, err := p.Predict(prompt, opts...)
	if err != nil {
		log.Printf("p.Predict() failed: %s\n", err)
		// panic on HTTP/1.x closes the connection,
		// on HTTP/2 it sends RST_STREAM,
		// so the client knows the stream ended prematurely
		panic(http.ErrAbortHandler)
	}
	log.Printf("<response>%s</response>\n", response)
}

type PredictHandler struct {
	Predictor predictor.Predictor
	StopRegex *regexp.Regexp
}

func (h PredictHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case "GET", "POST":
		err := r.ParseForm()
		if err != nil {
			w.WriteHeader(http.StatusBadRequest)
			fmt.Fprintf(w, http.StatusText(http.StatusBadRequest))
			return
		}
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		prompt := r.Form.Get("prompt")
		handlePrediction(w, r, h.Predictor, prompt, h.StopRegex)
	default:
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		w.WriteHeader(http.StatusMethodNotAllowed)
		fmt.Fprintf(w, "only GET and POST methods supported")
		return
	}
}

type ChatHandler struct {
	Predictor      predictor.Predictor
	PromptTemplate conversation.PromptTemplate
	SystemPrompt   string
	StopRegex      *regexp.Regexp
}

func (h ChatHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case "GET", "POST":
		err := r.ParseForm()
		if err != nil {
			w.WriteHeader(http.StatusBadRequest)
			fmt.Fprintf(w, http.StatusText(http.StatusBadRequest))
			return
		}
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		systemPrompt := h.SystemPrompt
		systemPromptGiven := r.Form.Get("system")
		if systemPromptGiven != "" {
			systemPrompt = systemPromptGiven
		}
		conv := conversation.NewConversation(systemPrompt)
		messages := r.Form["messages"]
		for i, message := range messages {
			if i%2 == 0 {
				conv.AddMessageUser(message)
			} else {
				conv.AddMessageAssistant(message)
			}
		}
		prompt, err := conv.GeneratePrompt(h.PromptTemplate)
		if err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			fmt.Fprintf(w, "conv.GeneratePrompt() failed: %s", err)
			return
		}
		replyPrefix := r.Form.Get("replyPrefix")
		if replyPrefix != "" {
			if !strings.HasSuffix(prompt, "\n") {
				prompt += " "
			}
			prompt += replyPrefix
		}
		handlePrediction(w, r, h.Predictor, prompt, h.StopRegex)
	default:
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		w.WriteHeader(http.StatusMethodNotAllowed)
		fmt.Fprintf(w, "only GET and POST methods supported")
		return
	}
}

type ModelConfig struct {
	GpuLayers              int     `json:"gpuLayers"`
	ContextSize            int     `json:"context"`
	PromptTemplate         string  `json:"promptTemplate"`
	PromptTemplateType     string  `json:"promptTemplateType"`
	PromptTemplateFilePath string  `json:"promptTemplateFile"`
	RopeFreqBase           float64 `json:"ropeFreqBase"`
	RopeFreqScale          float64 `json:"ropeFreqScale"`
}

type PredictConfig struct {
	Threads              int
	Tokens               int
	SystemPrompt         string
	SystemPromptFilePath string
	StopRegex            string
	NKeep                int
	TopK                 int
	TopP                 float64
	Temperature          float64
	TailFreeSamplingZ    float64
	RepetitionPenalty    float64
	FrequencyPenalty     float64
	PresencePenalty      float64
	Mirostat             int
	MirostatTau          float64
	MirostatEta          float64
}

type Config struct {
	Model               ModelConfig
	ModelConfigFilePath string
	Predict             PredictConfig
	Addr                string
	License             bool
}

func main2() error {
	var config Config

	// HTTP server options
	flag.StringVar(&config.Addr, "addr", "localhost:8080", `TCP network address the server listens on, in the form "host:port" or ":port" (e.g. "localhost:8080" or "127.0.0.1:8080" or ":8080")`)

	// Model options
	flag.IntVar(&config.Model.ContextSize, "context", 512, "context size")
	flag.IntVar(&config.Model.GpuLayers, "gpu-layers", 0, "number of GPU layers")
	flag.StringVar(&config.Model.PromptTemplate, "prompt-template", "", "prompt template. Setting the prompt template with this or the other prompt template flags is required if you want to use the /chat API endpoint")
	flag.StringVar(&config.Model.PromptTemplateFilePath, "prompt-template-file", "", "path to prompt template file. Setting the prompt template with this or the other prompt template flags is required if you want to use the /chat API endpoint")
	flag.StringVar(&config.Model.PromptTemplateType, "prompt-template-type", "", "prompt template type. valid values: llama-2, vicuna_v1.1. Setting the prompt template with this or the other prompt template flags is required if you want to use the /chat API endpoint")
	flag.Float64Var(&config.Model.RopeFreqBase, "rope-freq-base", 0, "RoPE base frequency (default 10000 unless specified in the GGUF file)")
	flag.Float64Var(&config.Model.RopeFreqScale, "rope-freq-scale", 0, "RoPE frequency scaling factor (default 1 unless specified in the GGUF file)")
	flag.StringVar(&config.ModelConfigFilePath, "model-config-file", "", "path to config file for the model")

	// Predict options
	flag.IntVar(&config.Predict.NKeep, "n-keep", 0, "number of tokens to keep from initial prompt (0 = disabled)")
	flag.StringVar(&config.Predict.StopRegex, "stop-regex", "", "regular expression that will stop prediction, if a match is found (experimental)")
	flag.StringVar(&config.Predict.SystemPrompt, "system-prompt", "", "system prompt")
	flag.StringVar(&config.Predict.SystemPromptFilePath, "system-prompt-file", "", "read the system prompt from this file")
	flag.IntVar(&config.Predict.Threads, "threads", runtime.NumCPU(), "number of threads")
	flag.IntVar(&config.Predict.Tokens, "tokens", 0, "number of tokens to predict (0 = no limit)")

	// Sampling options
	flag.IntVar(&config.Predict.TopK, "top-k", 40, "top-k")
	flag.Float64Var(&config.Predict.TopP, "top-p", 0.2, "top-p (1 = disabled)")
	flag.Float64Var(&config.Predict.Temperature, "temperature", 0.8, "temperature")
	flag.Float64Var(&config.Predict.TailFreeSamplingZ, "tail-free-sampling-z", 1, "tail free sampling parameter z (1 = disabled)")
	flag.Float64Var(&config.Predict.FrequencyPenalty, "penalty-frequency", 0.1, "frequency penalty (0 = disabled)")
	flag.Float64Var(&config.Predict.PresencePenalty, "penalty-presence", 0, "presense penalty (0 = disabled)")
	flag.Float64Var(&config.Predict.RepetitionPenalty, "penalty-repetition", 1.1, "repetition penalty (1 = disabled)")
	flag.IntVar(&config.Predict.Mirostat, "mirostat", 0, "mirostat (0 = disabled, 1 = mirostat, 2 = mirostat 2.0)")
	flag.Float64Var(&config.Predict.MirostatTau, "mirostat-tau", 5, "mirostat target entropy")
	flag.Float64Var(&config.Predict.MirostatEta, "mirostat-eta", 0.1, "mirostat learning rate")

	// other options
	flag.BoolVar(&config.License, "license", false, "show license")

	flag.Parse()

	if config.License {
		fmt.Printf("%v\n", license)
		for _, licenseDep := range licenseDeps {
			fmt.Printf("\n%v\n", licenseDep)
		}
		return nil
	}

	args := flag.Args()
	if len(args) == 0 {
		return errors.New("no arguments")
	}
	if len(args) > 1 {
		return errors.New("too many arguments")
	}

	if config.ModelConfigFilePath != "" {
		modelConfigFile, err := os.Open(config.ModelConfigFilePath)
		if err != nil {
			return fmt.Errorf("failed to open model config file: %s", err)
		}
		err = json.NewDecoder(modelConfigFile).Decode(&config.Model)
		if err != nil {
			return fmt.Errorf("failed to parse model config file: %s", err)
		}
		modelConfigFile.Close()
		// parse flags again because command line options have priority over config file
		flag.Parse()
	}

	var stopRegex *regexp.Regexp
	if config.Predict.StopRegex != "" {
		var err error
		stopRegex, err = regexp.Compile(config.Predict.StopRegex)
		if err != nil {
			return fmt.Errorf("failed to parse regex of flag -stop-regex: %s", err)
		}
	}

	// set systemPrompt
	var systemPrompt string
	if config.Predict.SystemPrompt != "" && config.Predict.SystemPromptFilePath != "" {
		return errors.New("cannot use flags -system-prompt and -system-prompt-file at the same time")
	}
	if config.Predict.SystemPrompt != "" {
		systemPrompt = config.Predict.SystemPrompt
	} else if config.Predict.SystemPromptFilePath != "" {
		systemPromptBytes, err := os.ReadFile(config.Predict.SystemPromptFilePath)
		if err != nil {
			return fmt.Errorf("failed to read system prompt from file: %s", err)
		}
		systemPrompt = strings.TrimSpace(string(systemPromptBytes))
	}

	// make sure only one of the -prompt-template* flags is set
	if config.Model.PromptTemplate != "" && config.Model.PromptTemplateType != "" {
		return errors.New("conflicting flags: -prompt-template -prompt-template-type")
	}
	if config.Model.PromptTemplate != "" && config.Model.PromptTemplateFilePath != "" {
		return errors.New("conflicting flags: -prompt-template -prompt-template-file")
	}
	if config.Model.PromptTemplateType != "" && config.Model.PromptTemplateFilePath != "" {
		return errors.New("conflicting flags: -prompt-template-type -prompt-template-file")
	}
	// set promptTemplate from one of the -prompt-template* flags
	var promptTemplate conversation.PromptTemplate
	if config.Model.PromptTemplate != "" {
		var err error
		promptTemplate, err = conversation.NewPromptTemplate(config.Model.PromptTemplate)
		if err != nil {
			return fmt.Errorf("failed to create prompt template: %s", err)
		}
	} else if config.Model.PromptTemplateType != "" {
		switch config.Model.PromptTemplateType {
		case "llama-2":
			promptTemplate = conversation.PromptTemplateLlama2
		case "vicuna_v1.1":
			promptTemplate = conversation.PromptTemplateVicunaV11
		default:
			return fmt.Errorf("invalid value of prompt_template_type: '%s'", config.Model.PromptTemplateType)
		}
	} else if config.Model.PromptTemplateFilePath != "" {
		if promptTemplate.Template != nil {
			return errors.New("cannot set both prompt_template_type and prompt_template_file")
		}
		promptTemplateFileBytes, err := os.ReadFile(config.Model.PromptTemplateFilePath)
		if err != nil {
			return fmt.Errorf("failed to read prompt template file '%s': %s", config.Model.PromptTemplateFilePath, err)
		}
		promptTemplateTemplate, err := template.New("user").Parse(string(promptTemplateFileBytes))
		if err != nil {
			return fmt.Errorf("failed to parse prompt template file: %s", err)
		}
		promptTemplate = conversation.PromptTemplate{
			Template: promptTemplateTemplate,
		}
	}

	configJSON, _ := json.MarshalIndent(config, "", "  ")
	fmt.Println(string(configJSON))

	// fail if system prompt is not set and it is required
	if systemPrompt == "" && promptTemplate.RequiresSystemPrompt {
		return errors.New("system prompt not set but the prompt template requires one")
	}

	modelFilePath := args[0]
	modelOptions := []llama.ModelOption{
		llama.SetContext(config.Model.ContextSize),
		llama.SetGPULayers(config.Model.GpuLayers),
	}
	if config.Model.RopeFreqBase != 0 {
		modelOptions = append(modelOptions, llama.WithRopeFreqBase(float32(config.Model.RopeFreqBase)))
	}
	if config.Model.RopeFreqScale != 0 {
		modelOptions = append(modelOptions, llama.WithRopeFreqScale(float32(config.Model.RopeFreqScale)))
	}
	predictor, err := predictor.New(
		modelFilePath,
		modelOptions,
		[]llama.PredictOption{
			llama.SetTokens(config.Predict.Tokens),
			llama.SetThreads(config.Predict.Threads),
			llama.SetNKeep(config.Predict.NKeep),
			llama.SetTopK(config.Predict.TopK),
			llama.SetTopP(float32(config.Predict.TopP)),
			llama.SetTemperature(float32(config.Predict.Temperature)),
			llama.SetTailFreeSamplingZ(float32(config.Predict.TailFreeSamplingZ)),
			llama.SetPenalty(float32(config.Predict.RepetitionPenalty)),
			llama.SetFrequencyPenalty(float32(config.Predict.FrequencyPenalty)),
			llama.SetPresencePenalty(float32(config.Predict.PresencePenalty)),
			llama.SetMirostat(config.Predict.Mirostat),
			llama.SetMirostatTAU(float32(config.Predict.MirostatTau)),
			llama.SetMirostatETA(float32(config.Predict.MirostatEta)),
			llama.SetPenalizeNL(false),
		},
	)
	if err != nil {
		return fmt.Errorf("predictor.New() failed: %s", err)
	}
	defer predictor.Free()

	mux := http.NewServeMux()
	mux.Handle("/predict", PredictHandler{
		Predictor: predictor,
		StopRegex: stopRegex,
	})
	if promptTemplate.Template != nil {
		mux.Handle("/chat", ChatHandler{
			Predictor:      predictor,
			PromptTemplate: promptTemplate,
			SystemPrompt:   systemPrompt,
			StopRegex:      stopRegex,
		})
	} else {
		log.Println("`/chat` endpoint is not working because prompt template is not set")
	}

	s := &http.Server{
		Handler:     mux,
		ReadTimeout: 30 * time.Second,
		Addr:        config.Addr,
	}
	err = s.ListenAndServe()
	if err != nil {
		return fmt.Errorf("ListenAndServe() failed: %s", err)
	}
	return nil
}

func main() {
	err := main2()
	if err != nil {
		fmt.Printf("FATAL ERROR: %s\n", err)
		os.Exit(1)
	}
}
