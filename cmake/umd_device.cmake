
if(TT_METAL_SIMULATOR_EN)
    add_library(simulation STATIC
        ${UMD_HOME}/device/simulation/tt_simulation_device.cpp
        ${UMD_HOME}/device/simulation/tt_simulation_host.cpp
    )
    target_link_libraries(simulation PUBLIC compiler_flags nng)
    target_include_directories(simulation PUBLIC
        ${UMD_HOME}
        ${UMD_HOME}/device
        ${UMD_HOME}/third_party/fmt/include
        ${flatbuffers_SOURCE_DIR}/include
        ${nanomsg_SOURCE_DIR}/include
    )
    if(DEFINED ENV{CONDA_PREFIX})
        target_include_directories(simulation PRIVATE $ENV{CONDA_PREFIX}/include)
    endif()
endif()

set(UMD_SRC
    ${UMD_HOME}/device/architecture_implementation.cpp
    ${UMD_HOME}/device/blackhole_implementation.cpp
    ${UMD_HOME}/device/cpuset_lib.cpp
    ${UMD_HOME}/device/grayskull_implementation.cpp
    ${UMD_HOME}/device/tlb.cpp
    ${UMD_HOME}/device/tt_cluster_descriptor.cpp
    ${UMD_HOME}/device/tt_device.cpp
    ${UMD_HOME}/device/tt_emulation_stub.cpp
    ${UMD_HOME}/device/tt_silicon_driver.cpp
    ${UMD_HOME}/device/tt_silicon_driver_common.cpp
    ${UMD_HOME}/device/tt_soc_descriptor.cpp
    ${UMD_HOME}/device/tt_versim_stub.cpp
    ${UMD_HOME}/device/wormhole_implementation.cpp
)

add_library(umd_device STATIC ${UMD_SRC})
target_include_directories(umd_device PRIVATE 
    ${UMD_HOME}
    ${UMD_HOME}/device
    ${UMD_HOME}/third_party/fmt/include)
target_link_libraries(umd_device PRIVATE yaml-cpp::yaml-cpp Boost::interprocess rt compiler_flags $<$<BOOL:${TT_METAL_SIMULATOR_EN}>:simulation>)

if(DEFINED ENV{CONDA_PREFIX})
    target_include_directories(umd_device PRIVATE $ENV{CONDA_PREFIX}/include)
endif()
