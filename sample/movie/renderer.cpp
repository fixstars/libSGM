/*
Copyright 2016 Fixstars Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "renderer.h"

Renderer::Renderer(int width, int height) : width_(width), height_(height) {
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	// init texture

	glGenTextures(1, &texture_);
	glBindTexture(GL_TEXTURE_2D, texture_);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);

	glGenTextures(1, &disp_texture_);
	glBindTexture(GL_TEXTURE_2D, disp_texture_);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_R16, width_, height_, 0, GL_RED, GL_UNSIGNED_SHORT, 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);

	// init shader program 
	static const char* vert_src =
		"#version 330 core\n"
		"layout(location = 0) in vec3 position;\n"
		"layout(location = 1) in vec2 tex_coord;\n"
		"out vec2 uv_coord;\n"
		"void main() {\n"
		"gl_Position = vec4(position.xyz, 1.0);\n"
		"uv_coord = tex_coord\n;"
		"}\n";

	static const char* input_frag_src =
		"#version 330 core\n"
		"precision highp float;\n"
		"out mediump vec4 fragColor;\n"
		"in vec2 uv_coord;\n"
		"uniform highp usampler2D tex_sampler;\n"
		"void main() {\n"
		"	float x = float(texture(tex_sampler, uv_coord).r) / 65535.0;\n"
		"	vec4 color = vec4(x, x, x, 1.0);\n"
		"	fragColor = color;\n"
		"}\n";
	program_input_ = compile_shader_program(vert_src, input_frag_src);

	static const char* disp_frag_src =
		"#version 330 core\n"
		"precision highp float;\n"
		"out mediump vec4 fragColor;\n"
		"in vec2 uv_coord;\n"
		"uniform sampler2D tex_sampler;\n"
		"uniform int inv_disp_size;\n"
		"void main() {\n"
		"	float x = texture(tex_sampler, uv_coord).r * inv_disp_size;\n"
		"	vec4 color = vec4(x, x, x , 1.0);\n"
		"	fragColor = color;\n"
		"}\n";
	program_disp_ = compile_shader_program(vert_src, disp_frag_src);

	static const char* cdisp_frag_src =
		"#version 330 core\n"
		"precision highp float;\n"
		"out mediump vec4 fragColor;\n"
		"in vec2 uv_coord;\n"
		"uniform sampler2D tex_sampler;\n"
		"uniform int inv_disp_size;\n"
		"void main() {\n"
		"	float val = texture(tex_sampler, uv_coord).r * inv_disp_size;\n"
		"	vec4 color = clamp(vec4(-2.0 + val * 4.0, 2.0 - abs(val-0.5) * 4.0, 2.0 - val * 4.0, 1.0), 0.0, 1.0);\n"
		"	fragColor = color;\n"
		"}\n";
	program_cdisp_ = compile_shader_program(vert_src, cdisp_frag_src);

	// init array buffer
	glGenBuffers(1, &vert_buffer_);
	glBindBuffer(GL_ARRAY_BUFFER, vert_buffer_);
	const float verts[] = { // x, y, z, u, v
		-1.0, 1.0, 0.0, 0.0, 0.0,
		-1.0, -1.0, 0.0, 0.0, 1.0,
		1.0, 1.0, 0.0, 1.0, 0.0,
		1.0, -1.0, 0.0, 1.0, 1.0,
	};
	glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

Renderer::~Renderer() {
	glDeleteTextures(1, &texture_);

	glDeleteProgram(program_input_);
	glDeleteProgram(program_disp_);
	glDeleteProgram(program_cdisp_);

	glDeleteBuffers(1, &vert_buffer_);
}

void Renderer::render_input(const uint16_t* h_input_ptr) {
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture_);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R16UI, width_, height_, 0, GL_RED_INTEGER, GL_UNSIGNED_SHORT, h_input_ptr);

	glUseProgram(program_input_);

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, vert_buffer_);

	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 5, (void*)0);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 5, (float*)0 + 3);
	GLint loc;
	loc = glGetUniformLocation(program_input_, "tex_sampler");
	if (loc != -1) {
		glUniform1i(loc, 0);
	}

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

	glBindTexture(GL_TEXTURE_2D, 0);
}

void Renderer::render_disparity(const uint16_t* d_disp, int disp_size) {
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, 0);
	
	// cuda-gl interop
	cudaGraphicsResource_t cuda_gl_tex_resource;
	cudaGraphicsGLRegisterImage(&cuda_gl_tex_resource, disp_texture_, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
	cudaGraphicsMapResources(1, &cuda_gl_tex_resource);
	cudaArray_t texture_array;
	cudaGraphicsSubResourceGetMappedArray(&texture_array, cuda_gl_tex_resource, 0, 0);
	
	cudaResourceDesc desc;
	desc.resType = cudaResourceTypeArray;
	desc.res.array.array = texture_array;

	cudaSurfaceObject_t write_surface;
	cudaCreateSurfaceObject(&write_surface, &desc);

	write_surface_U16_with_multiplication(write_surface, d_disp, width_, height_, 256);

	cudaDestroySurfaceObject(write_surface);

	cudaGraphicsUnmapResources(1, &cuda_gl_tex_resource);
	cudaGraphicsUnregisterResource(cuda_gl_tex_resource);
	// end cuda-gl interop

	glUseProgram(program_disp_);
	glBindTexture(GL_TEXTURE_2D, disp_texture_);
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, vert_buffer_);

	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 5, (void*)0);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 5, (float*)0 + 3);
	GLint loc;
	loc = glGetUniformLocation(program_disp_, "tex_sampler");
	if (loc != -1) {
		glUniform1i(loc, 0);
	}
	loc = glGetUniformLocation(program_cdisp_, "inv_disp_size");
	if (loc != -1) {
		glUniform1i(loc, 256 / disp_size);
	}

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

	glBindTexture(GL_TEXTURE_2D, 0);
}

void Renderer::render_disparity_color(const uint16_t* d_disp, int disp_size) {
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, 0);

	// cuda-gl interop
	cudaGraphicsResource_t cuda_gl_tex_resource;
	cudaGraphicsGLRegisterImage(&cuda_gl_tex_resource, disp_texture_, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
	cudaGraphicsMapResources(1, &cuda_gl_tex_resource);
	cudaArray_t texture_array;
	cudaGraphicsSubResourceGetMappedArray(&texture_array, cuda_gl_tex_resource, 0, 0);

	cudaResourceDesc desc;
	desc.resType = cudaResourceTypeArray;
	desc.res.array.array = texture_array;

	cudaSurfaceObject_t write_surface;
	cudaCreateSurfaceObject(&write_surface, &desc);

	write_surface_U16_with_multiplication(write_surface, d_disp, width_, height_, 256);

	cudaDestroySurfaceObject(write_surface);

	cudaGraphicsUnmapResources(1, &cuda_gl_tex_resource);
	cudaGraphicsUnregisterResource(cuda_gl_tex_resource);
	// end cuda-gl interop

	glUseProgram(program_cdisp_);
	glBindTexture(GL_TEXTURE_2D, disp_texture_);
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, vert_buffer_);

	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 5, (void*)0);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 5, (float*)0 + 3);
	GLint loc;
	loc = glGetUniformLocation(program_cdisp_, "tex_sampler");
	if (loc != -1) {
		glUniform1i(loc, 0);
	}
	loc = glGetUniformLocation(program_cdisp_, "inv_disp_size");
	if (loc != -1) {
		glUniform1i(loc, 256 / disp_size);
	}
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

	glBindTexture(GL_TEXTURE_2D, 0);
}


// util func
GLuint compile_shader_program(const char *vertex_shader_src, const char *fragment_shader_src) {
	GLuint program = glCreateProgram();
	GLuint vert, frag;
	if (vertex_shader_src) {
		vert = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vert, 1, &vertex_shader_src, NULL);
		glCompileShader(vert);

		GLint compiled = 0;
		glGetShaderiv(vert, GL_COMPILE_STATUS, &compiled);

		if (!compiled) {
			char tmp[256] = "";
			glGetShaderInfoLog(vert, 256, NULL, tmp);
			printf("vertex shader error:\n%s\n", tmp);
			glDeleteShader(vert);
			return 0;
		}
	}

	if (fragment_shader_src) {
		frag = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(frag, 1, &fragment_shader_src, NULL);
		glCompileShader(frag);

		GLint compiled = 0;
		glGetShaderiv(frag, GL_COMPILE_STATUS, &compiled);

		if (!compiled) {
			char tmp[256] = "";
			glGetShaderInfoLog(frag, 256, NULL, tmp);
			printf("fragment shader error:\n%s\n", tmp);
			glDeleteShader(frag);
			return 0;
		}
	}
	glAttachShader(program, vert);
	glAttachShader(program, frag);

	glLinkProgram(program);

	GLint linked = 0;
	glGetProgramiv(program, GL_LINK_STATUS, &linked);

	if (linked == 0) {
		int info_length = 0;
		glGetProgramiv(program, GL_INFO_LOG_LENGTH, (GLint *)&info_length);
		if (info_length > 0) {
			char* log = (char *)malloc(info_length);
			GLsizei dammy = 0;
			glGetProgramInfoLog(program, info_length, &dammy, log);
			printf("Shader compilation error: %s\n", log);
			free(log);
		}
	}
	glDeleteShader(vert);
	glDeleteShader(frag);

	return program;
}


