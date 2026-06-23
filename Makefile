BUILD_DIR ?= build

.PHONY: all test clean install docs

all:
	cmake --build $(BUILD_DIR)

test:
	cd $(BUILD_DIR) && ctest --output-on-failure

clean:
	cmake --build $(BUILD_DIR) --target clean

install:
	cmake --install $(BUILD_DIR)

docs:
	cmake --build $(BUILD_DIR) --target docs
