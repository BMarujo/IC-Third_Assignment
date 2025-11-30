CXX = g++
CXXFLAGS = -O3 -Wall
LDFLAGS = -lzstd

TARGET = compressor
SRC = main.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET) model.safetensors.zst model_restored.safetensors
