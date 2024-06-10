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

add_library(umd_device OBJECT ${UMD_SRC})
target_include_directories(umd_device PRIVATE ${UMD_HOME} ${UMD_HOME}/third_party/fmt/include)

foreach(lib ${BoostPackages})
    target_include_directories(umd_device PRIVATE ${Boost${lib}_SOURCE_DIR}/include)
endforeach()

target_link_libraries(umd_device PUBLIC yaml-cpp::yaml-cpp rt compiler_flags)
