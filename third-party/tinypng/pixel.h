/*!
 * @file pixel.h
 * Definition of the Pixel class for TinyPNG.
 * 
 * @author Rahul A. G.
 */
#ifndef __TINY_PIXEL_H__
#define __TINY_PIXEL_H__

// C++ includes
#include <stdint.h>

namespace tinypng
{
	/*!
	 * Represents a single 32-bit pixel.
	 */
	class Pixel
	{
	public:
		Pixel(uint8_t *data);
		uint8_t& red();
		uint8_t& green();
		uint8_t& blue();
		uint8_t& alpha();
		bool operator==(Pixel const& other);
		bool operator!=(Pixel const& other);		
	private:
		uint8_t *_data;
	};
}

#endif
