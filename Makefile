CGO_LDFLAGS_=
CGO_LDFLAGS_openblas=-lopenblas
CGO_LDFLAGS_clblas=-lOpenCL -lclblast -L/usr/local/lib64/
CGO_LDFLAGS:=$(CGO_LDFLAGS_$(BUILD_TYPE))

GO_BUILD_TAGS_=$(GO_BUILD_TAGS)
GO_BUILD_TAGS_openblas=$(GO_BUILD_TAGS_)openblas
GO_BUILD_TAGS_clblas=$(GO_BUILD_TAGS_)
GO_BUILD_TAGS:=$(GO_BUILD_TAGS_$(BUILD_TYPE))

.PHONY: all clean

all: llm-api

go-llama/libbinding.a:
	$(MAKE) -C go-llama.cpp BUILD_TYPE=$(BUILD_TYPE) libbinding.a

llm-api: go-llama/libbinding.a
	CGO_LDFLAGS="$(CGO_LDFLAGS)" go build -o llm-api -trimpath -tags "$(GO_BUILD_TAGS)"

clean:
	$(MAKE) -C go-llama.cpp clean
	rm -f ./llm-api
