# llm-api

*llm-api* offers a simple HTTP API that enables applications to perform inference with *LLMs*.

*llm-api* depends on *llama.cpp*, so it supports only models in *GGUF* format.

Under development. Not recommended for production use.

## HTTP API

Requests are in `x-www-form-urlencoded`, responses in plain text.

### Endpoints

#### `/predict` (GET or POST)

Submit a prompt to this endpoint and receive the response in plain text.

##### Query Parameters

- `prompt` (required)
- `stopRegex` (optional, experimental) regular expression that will stop prediction, if a match is found
- `temperature` (optional)

##### Returns

The response in plain text

##### Example Request

```sh
curl -X POST "http://localhost:8080/predict" -d "prompt=[INST] <<SYS>>
You are a helpful, respectful and honest assistant.
<</SYS>>

Hello! [/INST]"
```

#### `/chat` (GET or POST)

Submit a conversation to this endpoint and receive the response in plain text.

The difference from `/predict` is that the client doesn't need to know how to generate the prompt.
The server generates the prompt from the messages the client submits according to the prompt template.

This endpoint is activated, if you set the prompt template by using one of the flags: `-prompt-tempate` `-prompt-tempate-type` `-prompt-tempate-file`

##### Query Parameters

- `messages` (required) The messages of the conversation.
Use it multiple times if you have multiple messages in the conversation.
The first message of the conversation should belong to the user, the second to the assistant, etc.
The last message should belong to the user.
- `stopRegex` (optional, experimental) regular expression that will stop prediction, if a match is found
- `temperature` (optional)

##### Returns

The response in plain text

##### Example Request

```sh
curl -X POST "http://localhost:8080/chat" -d "messages=Hello" -d "messages=Hello! How can I help you?" -d "messages=Who are you?"
```

### Errors

#### Errors before inference starts

The server sends HTTP response with status code indicating an error has happened, and the description of the error in plain text.

#### Errors during inference

During inference, the server starts streaming the tokens to the client, and the status code is 200.
Since it's not possible to change the status code retroactively, if an error happens during inference, the server notifies the client about the error by closing the connection on HTTP/1.x, or by sending `RST_STREAM` on HTTP/2.
In both cases, the client knows the connection/stream was terminated because of an unknown error.
No information about the error is sent to the client.

## Usage

Download a model in *GGUF* format (e.g. from [TheBloke](https://huggingface.co/TheBloke)), and run one the following commands:

For the *Llama 2* model:
```sh
./llm-api -prompt-template-type llama-2 -context 4096 /path/to/model
```

Currently `llama-2` and `vicuna_v1.1` are the only built-in prompt templates,
but you can run other models, if you provide [your own template file](#custom-prompt-template) like this:
```sh
./llm-api -prompt-template-file /path/to/template -context 4096 /path/to/model
```

If the prompt template requires a system prompt, use the `-system-prompt` flag like this:
```sh
./llm-api -system-prompt "You are a helpful assistant" -prompt-template-type llama-2 /path/to/model
```

By default the HTTP server listens on `localhost:8080`. You can change this with the `-addr` flag:
```sh
./llm-api -addr ":80"
```

### Command line options
```
  -addr string
        TCP network address the server listens on, in the form "host:port" or ":port" (e.g. "localhost:8080" or "127.0.0.1:8080" or ":8080") (default "localhost:8080")
  -context int
        context size (default 512)
  -gpu-layers int
        number of GPU layers
  -mirostat int
        mirostat (0 = disabled, 1 = mirostat, 2 = mirostat 2.0)
  -mirostat-eta float
        mirostat learning rate (default 0.1)
  -mirostat-tau float
        mirostat target entropy (default 5)
  -model-config-file string
        path to config file for the model
  -n-keep int
        number of tokens to keep from initial prompt (0 = disabled)
  -penalty-frequency float
        frequency penalty (0 = disabled) (default 0.1)
  -penalty-presence float
        presense penalty (0 = disabled)
  -penalty-repetition float
        repetition penalty (1 = disabled) (default 1.1)
  -prompt-template string
        prompt template. Setting the prompt template with this or the other prompt template flags is required if you want to use the /chat API endpoint
  -prompt-template-file string
        path to prompt template file. Setting the prompt template with this or the other prompt template flags is required if you want to use the /chat API endpoint
  -prompt-template-type string
        prompt template type. valid values: llama-2, vicuna_v1.1. Setting the prompt template with this or the other prompt template flags is required if you want to use the /chat API endpoint
  -rope-freq-base float
        RoPE base frequency (default 10000 unless specified in the GGUF file)
  -rope-freq-scale float
        RoPE frequency scaling factor (default 1 unless specified in the GGUF file)
  -stop-regex value
        regular expression that will stop prediction, if a match is found (experimental)
  -system-prompt string
        system prompt
  -system-prompt-file string
        read the system prompt from this file
  -tail-free-sampling-z float
        tail free sampling parameter z (1 = disabled) (default 1)
  -temperature float
        temperature (default 0.8)
  -threads int
        number of threads (default number of CPU cores)
  -tokens int
        number of tokens to predict (0 = no limit)
  -top-k int
        top-k (default 40)
  -top-p float
        top-p (1 = disabled) (default 0.2)
```

### Custom Prompt Template

An example for chat LLM:
```
{{define "prompt"}}{{.SystemPrompt}}
{{range .MessagesWithoutSystemPrompt -}}
{{if eq .Role "user" }}
USER: {{.Text}}
{{else if eq .Role "assistant" }}
ASSISTANT: {{.Text}}
{{end}}{{end}}
ASSISTANT: {{end}}
```

For instruction LLMs, if you don't need chat functionality, you can simplify the prompt like this:
```
{{define "prompt"}}A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {{.LastMessageOfUser}} ASSISTANT: {{end}}
```

## Installation

You can install *llm-api* by building it from source.

Requirements: *Go 1.21*, *Git*, *make*, a *C* compiler
```sh
git clone --recurse-submodules https://github.com/cmitsakis/llm-api.git
cd llm-api
make
```

or if you want to build with *OpenBLAS*:
```sh
BUILD_TYPE=openblas make
```

or if you want to build with *OpenCL*:
```sh
BUILD_TYPE=clblas make
```

Installation via `go install` doesn't work because of the `go-llama.cpp` dependency which requires the use of Git submodules.

## Contributing

Bug reports, bug fixes, and performance improvements are welcome.
New features should be discussed first.

## License

Copyright (c) 2023 Charalampos Mitsakis <github.com/cmitsakis/llm-api>

Licensed under the [EUPL-1.2](LICENSE) or later.
