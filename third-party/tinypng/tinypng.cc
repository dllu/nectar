/*!
 * @file png.cc
 * Implementation of the TinyPNG C++ wrapper for libPNG.
 *
 * @author Rahul A. G.
 */

#include "tinypng.h"

using namespace tinypng;

inline void pngErr(string const& error)
{
	cerr << "[TinyPNG]: " << error << endl;
}

PNG::PNG()
{
	_buffer = NULL;
	_ext_buffer = false;
	_init();
}

PNG::PNG(int width, int height)
{
	_width = width;
	_height = height;
	_ext_buffer = false;
	_buffer = new uint8_t[_height * _width * PNG::BPP];
}

PNG::PNG(int width, int height, uint8_t *buffer)
{
	_width = width;
	_height = height;
	_ext_buffer = true;
	_buffer = buffer;
}

PNG::PNG(string const& file_name)
{
	_buffer = NULL;
	_ext_buffer = false;
	readFromFile(file_name);
}

PNG::PNG(PNG const& other)
{
	_copy(other);
}

PNG::~PNG()
{
	if (!_ext_buffer)
	{
		delete[] _buffer;
		_buffer = NULL;
	}
}

PNG const& PNG::operator=(PNG const& other)
{
	if (this != &other)
	{
		delete[] _buffer;
		_buffer = NULL;
		_copy(other);
	}
	return *this;
}

bool PNG::operator==(PNG const& other) const
{
	if (_width != other._width || _height != other._height)
	{
		return false;
	}

	return (memcmp(_buffer, other._buffer, _width * _height) == 0);
}

bool PNG::operator!=(PNG const & other) const
{
	return !(*this == other);
}

Pixel PNG::operator()(int x, int y)
{
	_clampXY(x, y);
	return _pixelAt(x,y);
}

bool PNG::readFromFile(string const& file_name)
{
	if (!_ext_buffer && _buffer != NULL)
	{
		delete[] _buffer;
		_buffer = NULL;
	}
	
	// we need to open the file in binary mode
	FILE * fp = fopen(file_name.c_str(), "rb");
	if (!fp)
	{
		pngErr("Failed to open " + file_name);
		return false;
	}

	// read in the header (max size of 8), use it to validate this as a PNG file
	png_byte header[8];
	fread(header, 1, 8, fp);
	if (png_sig_cmp(header, 0, 8))
	{
		pngErr("File is not a valid PNG file");
		fclose(fp);
		_init();
		return false;
	}

	// set up libpng structs for reading info
	png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL); if (!png_ptr)
	{
		pngErr("Failed to create read struct");
		fclose(fp);
		_init();
		return false;
	}

	png_infop info_ptr = png_create_info_struct(png_ptr);
	if (!info_ptr)
	{
		pngErr("Failed to create info struct");
		png_destroy_read_struct(&png_ptr, NULL, NULL);
		fclose(fp);
		_init();
		return false;
	}

	// set error handling to not abort the entire program
	if (setjmp(png_jmpbuf(png_ptr)))
	{
		pngErr("Error initializing libpng io");
		png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
		fclose(fp);
		_init();
		return false;
	}

	// initialize png reading
	png_init_io(png_ptr, fp);
	// let it know we've already read the first 8 bytes
	png_set_sig_bytes(png_ptr, 8);

	// read in the basic image info
	png_read_info(png_ptr, info_ptr);

	// If our PNG isn't in RGBA format, convert it
	png_set_expand(png_ptr);

	// convert to 8 bits
	png_byte bit_depth = png_get_bit_depth(png_ptr, info_ptr);
	if (bit_depth == 16)
		png_set_strip_16(png_ptr);

	if (!_ext_buffer)
	{
		_width = png_get_image_width(png_ptr, info_ptr);
		_height = png_get_image_height(png_ptr, info_ptr);
		_buffer = new uint8_t[_height * _width * PNG::BPP];
	}
	else
	{
		if (_width != png_get_image_width(png_ptr, info_ptr)
		 || _height != png_get_image_height(png_ptr, info_ptr))
		{
			pngErr("Image dimensions do not match external buffer");
			png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
			fclose(fp);
			return false;
		}
	}

	png_read_update_info(png_ptr, info_ptr);

	// begin reading in the image
	if (setjmp(png_jmpbuf(png_ptr)))
	{
		pngErr("Error reading image with libpng");
		png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
		fclose(fp);
		_init();
		return false;
	}


	int bpr = png_get_rowbytes(png_ptr, info_ptr); // number of bytes in a row

	// initialie our image storage
	png_byte * row = new png_byte[bpr];
	for (int y = 0; y < _height; y++)
	{
		png_read_row(png_ptr, row, NULL);
		memcpy(_buffer + y*_width*PNG::BPP, row, _width*PNG::BPP);
	}

	// cleanup
	delete[] row;
	png_read_end(png_ptr, NULL);
	png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
	fclose(fp);

	return true;
}

bool PNG::writeToFile(string const & file_name)
{
	FILE * fp = fopen(file_name.c_str(), "wb");
	if (!fp)
	{
		pngErr("Failed to open file " + file_name);
		return false;
	}

	png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (!png_ptr)
	{
		pngErr("Failed to create png struct");
		fclose(fp);
		return false;
	}

	png_infop info_ptr = png_create_info_struct(png_ptr);
	if (!info_ptr)
	{
		pngErr("Failed to create png info struct");
		png_destroy_write_struct(&png_ptr, NULL);
		fclose(fp);
		return false;
	}

	if (setjmp(png_jmpbuf(png_ptr)))
	{
		pngErr("Error initializing libpng io");
		png_destroy_write_struct(&png_ptr, &info_ptr);
		fclose(fp);
		return false;
	}

	png_init_io(png_ptr, fp);

	// Write header
	if (setjmp(png_jmpbuf(png_ptr)))
	{
		pngErr("Error writing image header");
		png_destroy_write_struct(&png_ptr, &info_ptr);
		fclose(fp);
		return false;
	}
	png_set_IHDR(png_ptr, info_ptr, _width, _height, 
			8,
			PNG_COLOR_TYPE_RGB_ALPHA, 
			PNG_INTERLACE_NONE, 
			PNG_COMPRESSION_TYPE_BASE,
			PNG_FILTER_TYPE_BASE);

	png_write_info(png_ptr, info_ptr);

	// Write image
	if (setjmp(png_jmpbuf(png_ptr)))
	{
		pngErr("Failed to write image");
		png_destroy_write_struct(&png_ptr, &info_ptr);
		fclose(fp);
		return false;
	}

	int bpr = png_get_rowbytes(png_ptr, info_ptr);
	png_byte * row = new png_byte[bpr];
	for (int y = 0; y < _height; y++)
	{
		memcpy(row, _buffer + y*_width*PNG::BPP, _width*PNG::BPP);
		png_write_row(png_ptr, row);
	}

	delete[] row;
	png_write_end(png_ptr, NULL);
	png_destroy_write_struct(&png_ptr, &info_ptr);
	fclose(fp);

	return true;
}

int PNG::getWidth() const
{
	return _width;
}

int PNG::getHeight() const
{
	return _height;
}

uint8_t *PNG::buffer() {
	return _buffer;
}

void PNG::_init()
{
	if (_ext_buffer)
	{
		return;
	}

	if (_buffer != NULL)
	{
		delete[] _buffer;
		_buffer = NULL;
	}
	_width = 1;
	_height = 1;
	_ext_buffer = false;
	_buffer = new uint8_t[1 * PNG::BPP];
	_blank();
}

void PNG::_blank()
{
	memset(_buffer, 0xFF, _width * _height * PNG::BPP);
}

void PNG::_copy(PNG const& other)
{
	if (!_ext_buffer)
	{
		_width = other._width;
		_height = other._height;
		_ext_buffer = false;
		_buffer = new uint8_t[_width * _height * PNG::BPP];
	} else {
		if (_width * _height != other._width * other._height)
		{
			return;
		}

		_width = other._width;
		_height = other._height;
	}
	
	memcpy(_buffer, other._buffer, _width * _height * PNG::BPP);
}

void PNG::_clampXY(int& x, int& y) const
{
	if (x < 0)
	{
		x = 0;
	}
	if (y < 0)
	{
		y = 0;
	}
	if (x > _width)
	{
		x = _width - 1;
	}
	if (y > _height)
	{
		y = _height - 1;
	}
}

Pixel PNG::_pixelAt(int x, int y) const
{
	return Pixel(_buffer + (y * _width + x));
}
