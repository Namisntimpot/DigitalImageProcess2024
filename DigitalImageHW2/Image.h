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
	// ϣ��ͼƬ���ݿ��Թ�����shared_ptr������Դ��������һ��Image������֮��ֱ�Ӱ�ͼƬ�ڴ��ͷ���.
	// ����Ϊrawdata�����Լ�����Ҳ������stb_image�����ĵ�.������Ҫָ��deleter
	//unsigned char* m_rawdata;
	std::shared_ptr<unsigned char[]> m_rawdata;
	int isNew;   // �Ƿ����´�����.

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
	// 0~255�Ҷȼ���ͼƬֱ��ͼ.
public:
	int channels, height, width;  // ���ֱ��ͼ��Ӧ��ͼƬ�Ŀ��ͨ��.
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