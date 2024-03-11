#include "Image.h"
#include <iostream>

// ------ColorU8-----------
ColorU8::ColorU8() : r(0), g(0), b(0), a(0) {};
ColorU8::ColorU8(unsigned char r, unsigned char g, unsigned char b, unsigned char a) :r(r), g(g), b(b), a(a) {};

// ------Image-------------
Image::Image(int h, int w, int channels) :height(h), width(w), channels(channels), isNew(1) {
	if (channels > 4) channels = 4;
	m_rawdata = new unsigned char[h * w * channels]();
}

Image::Image(const char* path): isNew(0) {
	m_rawdata = stbi_load(path, &width, &height, &channels, 0);
	if (m_rawdata == NULL) {
		std::cout << "Image Loading Failed";
	}
}

Image::Image(const string& path) : Image(path.c_str()) {};

Image::~Image() {
	if (isNew)
		delete[] m_rawdata;
	else
		stbi_image_free(m_rawdata);
}

ColorU8 Image::GetPixel(int h, int w) {
	if (!m_rawdata)
		throw std::runtime_error("Void Image.");
	int index = (h * width + w) * channels;
	int r = m_rawdata[index];
	int g = channels >= 2 ? m_rawdata[index + 1] : 0;
	int b = channels >= 3 ? m_rawdata[index + 2] : 0;
	int a = channels >= 4 ? m_rawdata[index + 3] : 0;
	return ColorU8(r, g, b, a);
}

void Image::SetPixel(ColorU8& color, int h, int w) {
	if (!m_rawdata)
		throw std::runtime_error("Void Image.");
	int index = (h * width + w) * channels;
	unsigned char tmp[] = { color.r, color.g, color.b, color.a };
	for (int i = 0; i < channels; ++i) {
		m_rawdata[index + i] = tmp[i];
	}
}

Image&& Image::PointTransformFromTable(unsigned char* table) {
	// table的格式是 256*channel
	Image img(height, width, channels);
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			int offset = (i * width + j) * channels;
			for (int k = 0; k < channels; ++k) {
				img.m_rawdata[offset + k] = table[m_rawdata[offset + k] * channels + k];
			}
		}
	}
	return std::move(img);
}

// 目前只支持输出png.
int Image::Write(const string& path) {
	string extension = path.substr(path.length() - 4);
	std::transform(extension.begin(), extension.end(), extension.begin(), std::tolower);

	if (extension == ".png")
		return stbi_write_png(path.c_str(), width, height, channels, m_rawdata, 0);
	else if (extension == ".jpg")
		return stbi_write_jpg(path.c_str(), width, height, channels, m_rawdata, 90);
	else
		return -1;
}


// Histogram
Histogram::Histogram(int h, int w, int channels) : channels(channels), height(h), width(w), m_level_map_table(NULL) {
	m_histdata = new unsigned int[MAX_COLOR_LEVEL_COUNT * channels]();  // 应该初始化为0了..?
}

Histogram::Histogram(Image& image) : channels(image.channels), height(image.height), width(image.height), m_level_map_table(NULL) {
	m_histdata = new unsigned int[MAX_COLOR_LEVEL_COUNT * image.channels]();
	for (int i = 0; i < image.height; ++i) {
		for (int j = 0; j < image.width; ++j) {
			// 这样遍历访问蛮低效的.
			ColorU8 color = image.GetPixel(i, j);
			unsigned char rgba[] = { color.r, color.g, color.b, color.a };
			for (int k = 0; k < image.channels; ++k) {
				m_histdata[rgba[k] * image.channels + k] += 1;
			}
		}
	}
}

Histogram::~Histogram() {
	delete[] m_histdata;
	if (m_level_map_table != NULL)
		delete[] m_level_map_table;
}

unsigned int Histogram::Get(int level, int channel) {
	return m_histdata[level * this->channels + channel];
}

void Histogram::Set(int level, int channel, unsigned int val) {
	m_histdata[level * this->channels + channel] = val;
}

unsigned char* Histogram::GetLevelMapTable() {
	return m_level_map_table;
}

void Histogram::Equalize() {
	unsigned int* cumulation = new unsigned int[MAX_COLOR_LEVEL_COUNT * channels]();
	for (int i = 0; i < channels; ++i) {
		cumulation[i] = m_histdata[i];
	}
	// 计算概率分布函数
	for (int i = 1; i < MAX_COLOR_LEVEL_COUNT; ++i) {
		int off = i * channels, last = (i-1) * channels;
		for (int j = 0; j < channels; ++j) {
			cumulation[off + j] = cumulation[last + j] + m_histdata[off + j];
		}
	}
	
	// 计算转换表.
	m_level_map_table = new unsigned char[MAX_COLOR_LEVEL_COUNT * channels];
	int hw = height * width;
	for (int i = 0; i < MAX_COLOR_LEVEL_COUNT * channels; ++i) {
		m_level_map_table[i] = static_cast<unsigned char>(std::round(static_cast<double>(cumulation[i]) / hw));
	}

	delete[] cumulation;
}