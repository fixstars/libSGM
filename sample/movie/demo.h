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

#pragma once

#include <stdint.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

class SGMDemo {
public:
	SGMDemo(int width, int height);
	~SGMDemo();

	int init();
	void swap_buffer();

	int should_close() { return glfwWindowShouldClose(this->window_); }
	uint32_t get_flag() const { return flag_; }

	void close();
private:
	int width_;
	int height_;
	GLFWwindow* window_;
	uint32_t flag_;

	friend void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
};
