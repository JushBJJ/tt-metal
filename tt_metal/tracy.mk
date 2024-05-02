TRACY_LIB = $(LIBDIR)/libtracy.so
TRACY_INCLUDES = -I$(TT_METAL_HOME)/tt_metal/third_party/tracy/public/tracy/
TRACY_LDFLAGS = $(LDFLAGS)
TRACY_DEFINES = -DTRACY_NO_CONTEXT_SWITCH
TRACY_BUILD_DIR = $(TT_METAL_HOME)/build/tools/profiler/bin
TRACY_REPO = $(TT_METAL_HOME)/tt_metal/third_party/tracy

#TRACY_DEFINES += -DTRACY_NO_SAMPLING -DTRACY_NO_SYSTEM_TRACING  -DTRACY_NO_CALLSTACK -DTRACY_NO_CALLSTACK_INLINES
TRACY_SRCS = \
	tt_metal/third_party/tracy/public/TracyClient.cpp

TRACY_OBJS = $(addprefix $(OBJDIR)/, $(TRACY_SRCS:.cpp=.o))
TRACY_DEPS = $(addprefix $(OBJDIR)/, $(TRACY_SRCS:.cpp=.d))

-include $(TRACY_DEPS)

tracy: $(TRACY_LIB)

$(TRACY_LIB): $(TRACY_OBJS)
	@mkdir -p $(LIBDIR)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(SHARED_LIB_FLAGS) -o $@ $^ $(TRACY_LDFLAGS)

$(OBJDIR)/tt_metal/third_party/tracy/public/%.o: tt_metal/third_party/tracy/public/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(TRACY_INCLUDES) $(TRACY_DEFINES) -c -o $@ $<

tracy_tools:
	mkdir -p $(TRACY_BUILD_DIR)
	cd $(TRACY_REPO)/csvexport/build/unix && $(MAKE)
	cp $(TRACY_REPO)/csvexport/build/unix/csvexport-release $(TRACY_BUILD_DIR)
	cd $(TRACY_REPO)/capture/build/unix && $(MAKE)
	cp $(TRACY_REPO)/capture/build/unix/capture-release $(TRACY_BUILD_DIR)

tracy_tools_clean:
	cd $(TRACY_REPO)/csvexport/build/unix && $(MAKE) clean
	cd $(TRACY_REPO)/capture/build/unix && $(MAKE) clean
