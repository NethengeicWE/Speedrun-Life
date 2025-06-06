# 自动构建系统介绍
## 引言
1. **什么是构建工具**：大型项目编译管理
1. **为什么需要它**：1k个cpp和h手动敲命令还是写脚本  
一般情况：$ gcc FuckMyLife.c -o FML

## 介绍
| 工具名          | 简介              | 优势                | 劣势                   | 适用场景            |
| ------------ | --------------- | ----------------- | -------------------- | --------------- |
| **CMake**    | 构建系统生成器         | 跨平台、社区庞大          | 配置语法复杂               | 推荐主流使用          |
| **Makefile** | 最基础的构建工具        | 简单直接、灵活           | 需手动维护依赖              | 小项目             |
| **Ninja**    | 快速构建执行器          | 极快、并行能力强          | 需其他工具生成构建脚本（如 CMake） | 速度敏感项目          |
| Meson    | 现代构建系统          | 简洁、高性能、原生支持 Ninja | 不如 CMake 兼容广泛        | 现代 C++ 项目，快速迭代  |
| Bazel    | Google 的构建系统    | 可扩展、高速、支持多语言      | 学习曲线高                | 大型、多语言项目        |
| Buck     | Facebook 构建系统   | 高度模块化、适合大型项目      | 不够通用，主要为 FB 内部优化     | 大型、分布式项目        |
| Qbs      | Qt 公司开发         | 支持复杂依赖管理          | 开发维护停滞（已废弃）          | Qt 项目（已不推荐）     |
| SCons    | 使用 Python 写构建脚本 | 可扩展性强             | 构建速度慢                | 教学、小型脚本化项目      |
| Premake  | 使用 Lua 编写构建配置   | 脚本灵活，可生成 IDE 工程   | 社区较小                 | 游戏开发等偏爱 Lua 的场景 |
| xmake    | 中国社区开发，支持 Lua   | 语法简单、快速构建、内建包管理   | 生态尚不如 CMake          | 小中型现代项目         |

## Makefile介绍：
基本编译工具，简单灵活，不够直观
1. 基本命令
```makefile
target: dependencies
	<tab> command
```
1. 变量定义与使用
```makefile
CC = g++
CFLAGS = -Wall -g
$(CC) $(CFLAGS) -c main.cpp
```
1. 不涉及文件生成的伪目标
```makefile
.PHONY: clean
clean:
	rm -f *.o main
```
1. 自动变量
```makefile
$@    # 代表规则的目标
$<    # 第一个依赖
$^    # 所有依赖
```
```makefile
main.o: main.cpp
	$(CC) -c $< -o $@
```
1. 条件判断

```makefile
ifeq ($(DEBUG), 1)
    CFLAGS += -g
else
    CFLAGS += -O2
endif
```
1. 






```makefile

# ========= 变量定义 =========

# 编译器
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Iinclude -Iinclude/freeglut/include -Iinclude/GLFW/include

# 输出目录
BIN_DIR = bin

# 源文件
SRC = src/main.cpp \
      include_custom/animation/Tree.cpp \
      include_custom/animation/Horse.cpp \
      include_custom/animation/MyScene.cpp \
      src/stb_image.cpp \
      src/glad.c

# 对应的中间目标文件（.o）
OBJ = $(SRC:.cpp=.o)
OBJ := $(OBJ:.c=.o)

# 链接库路径（使用 Windows 风格的路径分隔符）
LDFLAGS = -Linclude/freeglut/lib/x64 -Linclude/GLFW/lib-vc2022

# 链接库（注意顺序）
LDLIBS = -lfreeglut -lglfw3 -lopengl32

# 最终可执行文件名
TARGET = $(BIN_DIR)/OpenGL_Template.exe

# DLL 路径
DLLS = include/freeglut/bin/x64/freeglut.dll include/GLFW/lib-vc2022/glfw3.dll

# ========= 规则定义 =========

# 默认目标
all: $(TARGET) copy-dlls

# 编译目标
$(TARGET): $(OBJ)
	@echo "Linking $(TARGET)..."
	@mkdir -p $(BIN_DIR)
	$(CXX) $(OBJ) $(LDFLAGS) $(LDLIBS) -o $@

# 通配规则：将 .cpp / .c 文件编译为 .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.c
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 拷贝 DLL 文件到输出目录
copy-dlls:
	@echo "Copying DLLs to $(BIN_DIR)..."
	@for dll in $(DLLS); do \
		cp -u $$dll $(BIN_DIR); \
	done

# 清理构建产物
clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all clean copy-dlls

```
## Cmake介绍：
生成makefile，生成器的生成器
```c
cmake_minimum_required(VERSION 3.5)
project(OpenGL_Template)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# Debug模式
if(DEBUG)
  message("Debug mode")
  add_compile_definitions(DEBUG_MODE)
else()
  message("Release mode")
endif()


# 头文件路径
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/freeglut/include
    ${CMAKE_SOURCE_DIR}/include/GLFW/include
)

# 链接库路径.lib
link_directories(
    ${CMAKE_SOURCE_DIR}/include/freeglut/lib/x64
    ${CMAKE_SOURCE_DIR}/include/GLFW/lib-vc2022

)

# 添加静态库（编译）
add_library(STB_IMAGE STATIC "src/stb_image.cpp")
add_library(GLAD STATIC "src/glad.c")

# 可执行文件
add_executable(OpenGL_Template
    src/main.cpp
    include_custom/animation/Tree.cpp
    include_custom/animation/Horse.cpp
    include_custom/animation/MyScene.cpp
)

# 链接库
target_link_libraries(OpenGL_Template
    STB_IMAGE
    GLAD
    freeglut.lib           
    glfw3.lib
    opengl32.lib
)

# 复制运行的DLL
add_custom_command(TARGET OpenGL_Template POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_SOURCE_DIR}/include/freeglut/bin/x64/freeglut.dll
        $<TARGET_FILE_DIR:OpenGL_Template>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_SOURCE_DIR}/include/GLFW/lib-vc2022/glfw3.dll
        $<TARGET_FILE_DIR:OpenGL_Template>

)

```