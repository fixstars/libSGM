#include "demo.h"

#include <stdio.h>

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, 1);
	}
	else if (key == GLFW_KEY_S && action == GLFW_PRESS) {
		SGMDemo* self = (SGMDemo*)glfwGetWindowUserPointer(window);
		self->flag_ += 1;
		if (self->flag_ >= 3) {
			self->flag_ = 0;
		}
	}
}

SGMDemo::SGMDemo(int width, int height) : flag_(2), width_(width), height_(height) {

}

SGMDemo::~SGMDemo() {
}

int SGMDemo::init() {
	// init GLFW & GLEW
	glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API); // GLFW_OPENGL_ES_API
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	if (!glfwInit()) return -1;

	this->window_ = glfwCreateWindow(width_, height_, "SGM Demo", NULL, NULL);
	if (!this->window_) {
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(this->window_);

	GLenum glew_err = glewInit();
	if (glew_err != GLEW_OK) {
		printf("%s\n", glewGetErrorString(glew_err));
		return -2;
	}
	// setup event handler
	glfwSetWindowUserPointer(this->window_, this);
	glfwSetKeyCallback(this->window_, key_callback);

	return 0;
}

void SGMDemo::swap_buffer() {
	glfwSwapBuffers(this->window_);
	glfwPollEvents();
}

void SGMDemo::close() {
	glfwTerminate();

}
