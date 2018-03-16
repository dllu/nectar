/*!
 * @file tinypng_test.cc
 * Test file for TinyPNG.
 *
 * @author Rahul A. G.
 */

#include "png.h"

#include <iostream>
#include <stdio.h>

using namespace tinypng;

int main(int argc, char const *argv[])
{

	PNG input("test_in.png");

	input.writeToFile("test_out.png");

	PNG test("test_out.png");
	
	std::cout << "TinyPNG IO + codec test: ";
	if (input == test)
	{
		std::cout << "SUCCESS" << endl;
	}
	else
	{
		std::cout << "FAILURE" << endl;
	}

	test(0,0).red() -= 7;
	test(0,0).green() -= 7;
	test(0,0).blue() -= 7;
	test(0,0).alpha() -= 7;

	std::cout << "TinyPNG pixel access test: ";
	uint32_t v_test, v_input;
	v_input =  input(0,0).red() + input(0,0).green() + input(0,0).blue() + input(0,0).alpha();
	v_test =  test(0,0).red() + test(0,0).green() + test(0,0).blue() + test(0,0).alpha();
	if (v_test == v_input - 28)
	{
		std::cout << "SUCCESS" << endl;
	}
	else
	{
		std::cout << "FAILURE" << endl;
	}

	uint8_t *backing_store = new uint8_t[1920 * 1080 * PNG::BPP];
	PNG bs_test(1920, 1080, backing_store);

	bs_test.readFromFile("test_in.png");

	std::cout << "TinyPNG external bytestream test: ";
	if (input == bs_test)
	{
		std::cout << "SUCCESS" << endl;
	}
	else
	{
		std::cout << "FAILURE" << endl;
	}

	std::cout << "TinyPNG bytewise write test: ";
	bs_test.buffer()[22] = 0x0D;
	if (bs_test.buffer()[22] == 0x0D)
	{
		std::cout << "SUCCESS" << endl;
	}
	else
	{
		std::cout << "FAILURE" << endl;
	}

	remove("test_out.png");

	delete[] backing_store;
	backing_store = NULL;

	return 0;
}
