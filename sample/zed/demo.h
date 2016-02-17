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
