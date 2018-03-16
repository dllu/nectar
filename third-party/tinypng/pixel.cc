/*!
 * @file pixel.cc
 * Implementation of the Pixel class for TinyPNG.
 * 
 * @author Rahul A. G.
 */

#include "pixel.h"

using namespace tinypng;

Pixel::Pixel(uint8_t *data)
{
	_data = data;
}

uint8_t& Pixel::red()
{
	return _data[0];
}

uint8_t& Pixel::green()
{
	return _data[1];
}

uint8_t& Pixel::blue()
{
	return _data[2];
}

uint8_t& Pixel::alpha()
{
	return _data[3];
}

bool Pixel::operator==(Pixel const& other)
{
	return _data[0] == other._data[0]
	    && _data[1] == other._data[1]
	    && _data[2] == other._data[2]
	    && _data[3] == other._data[3];
}

bool Pixel::operator!=(Pixel const& other)
{
	return !(*this == other);
}
