#pragma once

#include<string>
#include<memory>
#include<algorithm>
#include<stdio.h>

#define __STDC_LIB_EXT1__

#define MAX_COLOR_LEVEL_VALUE 255
#define MAX_COLOR_LEVEL_COUNT 256

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
	// 希望图片数据可以共享，用shared_ptr进行资源管理，避免一个Image类析构之后直接把图片内存释放了.
	// 但因为rawdata可能自己分配也可能由stb_image创建的的.所以需要指定deleter
	//unsigned char* m_rawdata;
	std::shared_ptr<unsigned char[]> m_rawdata;
	int isNew;   // 是否是新创建的.

public:
	Image(int h, int w, int channels);
	Image(const char* path);
	Image(const string& path);
	~Image();

	ColorU8 GetPixel(int h, int w);
	void SetPixel(ColorU8& color, int h, int w);
	Image PointTransformFromTable(unsigned char* table);
	Image PointTransformLinear(int lorigin, int rorigin, int ltarget, int rtarget);
	Image ConvertToGray();
	std::shared_ptr<unsigned char[]> GetRawDataPtr();

	int Write(const string& path);
};

class Histogram {
	// 0~255灰度级的图片直方图.
public:
	int channels, height, width;  // 这个直方图对应的图片的宽高通道.
private:
	/*unsigned int* m_histdata;
	unsigned char* m_level_map_table;*/
	std::shared_ptr<unsigned int[]> m_histdata;
	std::shared_ptr<unsigned char[]> m_level_map_table;
public:
	Histogram(int h, int w, int channels);
	Histogram(Image& image);
	~Histogram();

	unsigned int Get(int level, int channel);
	void Set(int level, int channel, unsigned int val);
	unsigned char* GetLevelMapTable();
	void Equalize();

	int Draw(const string& path);
};