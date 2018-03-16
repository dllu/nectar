/*!
 * @file png.h
 * Definition of the TinyPNG C++ wrapper for libPNG.
 * 
 * @author Rahul A. G.
 */

#ifndef __TINY_PNG_H__
#define __TINY_PNG_H__ value

// C++ includes
#include <string>
#include <iostream>
#include <sstream>

// C includes
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <png.h>

// Local includes
#include "pixel.h"

using std::cerr;
using std::endl;
using std::string;
using std::stringstream;

namespace tinypng
{
	
	/*!
	 * Represents a PNG image.
	 */
	class PNG
	{
	public:

		/*!
		 * Creates a 1px x 1px PNG image.
		 */
		PNG();

		/*!
		 * Creates a PNG image of the desired dimensions.
		 *
		 * @param width Width of the new image.
		 * @param height Height of the new image.
		 */
		PNG(int width, int height);

		/*!
		 * Creates a PNG image of the desired dimensions using the buffer specified.
		 * Note: behaviour when sizeof(buffer) < width*height*PNG::BPP is undefined.
		 *
		 * @param width Width of the new image.
		 * @param height Height of the new image.
		 * @param buffer External buffer to use.
		 */
		PNG(int width, int height, uint8_t *buffer);

		/*!
		 * Creates a PNG image from the specified file on disk.
		 *
		 * @param file_name Name of the file to be read in.
		 */
		PNG(string const& file_name);

		/*!
		 * Copy constructor.
		 *
		 * @param other PNG to be copied.
		 */
		PNG(PNG const& other);

		/*!
		 * Destructor
		 */
		~PNG();

		/*!
		 * Assignment operator.
		 *
		 * @param other Image to copy.
		 * @return The current image, for assignment chaining.
		 */
		PNG const& operator=(PNG const& other);

		/*!
		 * Equality operator.
		 *
		 * @param other Image to be checked.
		 * @return Whether the current image is equal to the other image.
		 */
		bool operator==(PNG const& other) const;

		/*!
		 * Inequality operator.
		 *
		 * @param other Image to be checked.
		 * @return Whether the current image is different from the other image.
		 */
		bool operator!=(PNG const& other) const;

		/*!
		 * Pixel access operator. (0,0) is the upper left pixel.
		 *
		 * @param x X-coordinate for the pixel.
		 * @param y Y-coordinate for the pixel.
		 * @return The pixel at the given coordinates.
		 */
		Pixel operator()(int x, int y);

		/*!
		 * Reads in a PNG object from a file. Overwrites any data in the PNG object.
		 * In the event of failure, the object's contents are undefined.
		 *
		 * @param file_name Name of the file to read from.
		 * @return Whether the image was read successfully.
		 */
		bool readFromFile(string const& file_name);

		/*!
		 * Writes a PNG object to a file.
		 *
		 * @param file_name Name of the file to write to.
		 * @return Whether the file was written successfully.
		 */
		bool writeToFile(string const& file_name);

		/*!
		 * Gets the width of the image.
		 *
		 * @return Width of the image.
		 */
		int getWidth() const;

		/*!
		 * Gets the height of the image.
		 *
		 * @return Height of the image.
		 */
		int getHeight() const;

		/*!
		 * Gets the buffer backing the image.
		 *
		 * @return Pointer to the image buffer.
		 */
		uint8_t *buffer();

		static const int BPP = 4;

	private:
		int _width;
		int _height;
		uint8_t *_buffer;
		bool _ext_buffer;

		// Helper functions
		void _init();
		void _blank();
		void _copy(PNG const& other);
		void _syncBytes();
		void _clampXY(int& width, int& height) const;
		Pixel _pixelAt(int x, int y) const;
	};

}

#endif
