#pragma once

#include<string>
#include<memory>

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#endif // !STB_IMAGE_IMPLEMENTATION

#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#endif // !STB_IMAGE_WRITE_IMPLEMENTATION

#define MAX_COLOR_LEVEL 255

using std::string;

struct ColorU8 {  // R8G8B8A8
	unsigned char r, g, b, a;
	ColorU8();
	ColorU8(unsigned char r, unsigned char g, unsigned char b, unsigned char a);
};

class Image {
public:
	int height, width, channels;

private:
	unsigned char* m_rawdata;
	int isNew;   // 是否是新创建的.

public:
	Image(int h, int w, int channels);
	Image(const char* path);
	Image(const string& path);
	~Image();

	ColorU8 GetPixel(int h, int w);
	void SetPixel(ColorU8& color, int h, int w);
	Image&& PointTransformFromTable(unsigned char* table);

	int Write(const char* path);
};

class Histogram {
	// 0~255灰度级的图片直方图.
public:
	int channels, height, width;  // 这个直方图对应的图片的宽高通道.
private:
	unsigned int* m_histdata;
	unsigned char* m_level_map_table;
public:
	Histogram(int h, int w, int channels);
	Histogram(Image& image);
	~Histogram();

	unsigned int Get(int level, int channel);
	void Set(int level, int channel, unsigned int val);
	unsigned char* GetLevelMapTable();
	void Equalize();
};