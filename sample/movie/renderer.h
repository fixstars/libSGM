#pragma once

#include "demo.h"

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

GLuint compile_shader_program(const char *vertex_shader_src, const char *fragment_shader_src);

void write_surface_U16_with_multiplication(cudaSurfaceObject_t dst_surface, const uint16_t* d_src, int width, int height, uint16_t scale);

class Renderer {
	
public:
	Renderer(int width, int height);
	~Renderer();

	void render_input(const uint16_t* h_input_ptr);

	void render_disparity(const uint16_t* d_disp, int disp_size);

	void render_disparity_color(const uint16_t* d_disp, int disp_size);

private:
	Renderer(const Renderer&);
	int width_;
	int height_;

	GLuint texture_;
	GLuint disp_texture_;
	GLuint program_input_;
	GLuint program_disp_;
	GLuint program_cdisp_;
	GLuint vert_buffer_;
};
