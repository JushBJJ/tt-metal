#include "tt_metal/host_api.hpp"
#include "tensor/tensor.hpp"
#include "tensor/tensor_impl.hpp"
#include "tensor/owned_buffer.hpp"
#include "tensor/owned_buffer_functions.hpp"
#include "common/bfloat16.hpp"
#include "constants.hpp"

#include "tt_numpy/functions.hpp"

#include <algorithm>
#include <functional>
#include <random>

using namespace tt;
using namespace tt_metal;
using namespace constants;


bool test_tensor_copy_semantics(Device *device, Host *host) {

    bool pass = true;
    std::array<uint32_t, 4> single_tile_shape = {1, 1, TILE_HEIGHT, TILE_WIDTH};

    // host tensor to host tensor copy constructor
    Tensor host_a = tt::numpy::random::random(single_tile_shape).to(Layout::TILE);
    Tensor host_a_copy = host_a;
    auto host_a_data = owned_buffer::get_as<bfloat16>(host_a);
    auto host_a_copy_data = owned_buffer::get_as<bfloat16>(host_a_copy);
    pass &= host_a_data == host_a_copy_data;

    // dev tensor to dev tensor copy constructor
    Tensor dev_a = tt::numpy::random::random(single_tile_shape).to(Layout::TILE).to(device);
    Tensor dev_a_copy = dev_a;
    auto dev_a_on_host = dev_a.to(host);
    auto dev_a_copy_on_host = dev_a_copy.to(host);
    auto dev_a_data = owned_buffer::get_as<bfloat16>(dev_a_on_host);
    auto dev_a_copy_data = owned_buffer::get_as<bfloat16>(dev_a_copy_on_host);
    pass &= dev_a_data == dev_a_copy_data;

    // host tensor updated with host tensor copy assignment
    Tensor host_c = tt::numpy::arange<bfloat16>(0, tt_metal::volume(single_tile_shape), 1).reshape(single_tile_shape).to(Layout::TILE);
    Tensor host_c_copy = tt::numpy::random::random(single_tile_shape).to(Layout::TILE);
    host_c_copy = host_c;
    auto host_c_data = owned_buffer::get_as<bfloat16>(host_c);
    auto host_c_copy_data = owned_buffer::get_as<bfloat16>(host_c_copy);
    pass &= host_c_data == host_c_copy_data;

    // host tensor updated with dev tensor copy assignment
    Tensor host_d_copy = tt::numpy::random::random(single_tile_shape).to(Layout::TILE);
    host_d_copy = dev_a;
    pass &= (host_d_copy.storage_type() == StorageType::DEVICE);
    auto host_d_copy_on_host = host_d_copy.to(host);
    auto host_d_copy_data = owned_buffer::get_as<bfloat16>(host_d_copy_on_host);
    pass &= dev_a_data == host_d_copy_data;

    // dev tensor updated with host tensor copy assignment
    Tensor host_e = tt::numpy::ones(single_tile_shape).to(Layout::TILE);
    Tensor dev_e_copy = tt::numpy::random::random(single_tile_shape).to(Layout::TILE).to(device);
    dev_e_copy = host_e;
    pass &= (dev_e_copy.storage_type() == StorageType::OWNED);
    auto host_e_data = owned_buffer::get_as<bfloat16>(host_e);
    auto dev_e_copy_data = owned_buffer::get_as<bfloat16>(dev_e_copy);
    pass &= host_e_data == dev_e_copy_data;

    // dev tensor updated with dev tensor copy assignment
    Tensor dev_b = tt::numpy::ones(single_tile_shape).to(Layout::TILE).to(device);
    Tensor dev_b_copy = tt::numpy::zeros(single_tile_shape).to(Layout::TILE).to(device);
    dev_b_copy = dev_b;
    pass &= (dev_b_copy.storage_type() == StorageType::DEVICE);
    auto dev_b_on_host = dev_b.to(host);
    auto dev_b_copy_on_host = dev_b_copy.to(host);
    auto dev_b_data = owned_buffer::get_as<bfloat16>(dev_b_on_host);
    auto dev_b_copy_data = owned_buffer::get_as<bfloat16>(dev_b_copy_on_host);
    pass &= dev_b_data == dev_b_copy_data;

    return pass;
}

bool test_tensor_move_semantics(Device *device, Host *host) {
    bool pass = true;
    std::array<uint32_t, 4> single_tile_shape = {1, 1, TILE_HEIGHT, TILE_WIDTH};

    auto random_tensor = tt::numpy::random::uniform(bfloat16(-1.0f), bfloat16(1.0f), single_tile_shape);
    auto bfloat_data = random_tensor.owned_storage().value().buffer;
    auto bfloat_data_view = owned_buffer::get_as<bfloat16>(bfloat_data);

    // host tensor to host tensor move constructor
    Tensor host_a = Tensor(OwnedStorage{bfloat_data}, single_tile_shape, DataType::BFLOAT16, Layout::TILE);
    Tensor host_a_copy = std::move(host_a);
    auto host_a_copy_data = owned_buffer::get_as<bfloat16>(host_a_copy);
    pass &= host_a_copy_data == bfloat_data_view;

    // dev tensor to dev tensor move constructor
    Tensor dev_a = Tensor(OwnedStorage{bfloat_data}, single_tile_shape, DataType::BFLOAT16, Layout::TILE).to(device);
    auto og_buffer_a = dev_a.buffer();
    Tensor dev_a_copy = std::move(dev_a);
    pass &= (dev_a.buffer() == nullptr and dev_a_copy.buffer() == og_buffer_a);
    auto dev_a_copy_on_host = dev_a_copy.to(host);
    auto dev_a_copy_data = owned_buffer::get_as<bfloat16>(dev_a_copy_on_host);
    pass &= dev_a_copy_data == bfloat_data_view;

    // host tensor updated with host tensor move assignment
    auto random_tensor_three = tt::numpy::random::uniform(bfloat16(-1.0f), bfloat16(1.0f), single_tile_shape);
    auto bfloat_data_three = random_tensor_three.owned_storage().value().buffer;
    Tensor host_c = Tensor(OwnedStorage{bfloat_data_three}, single_tile_shape, DataType::BFLOAT16, Layout::TILE);
    Tensor host_c_copy = Tensor(dev_a_copy_on_host.owned_storage().value(), single_tile_shape, DataType::BFLOAT16, Layout::TILE);
    host_c_copy = std::move(host_c);
    auto host_c_copy_data = owned_buffer::get_as<bfloat16>(host_c_copy);
    pass &= host_c_copy_data == owned_buffer::get_as<bfloat16>(bfloat_data_three);;

    // host tensor updated with dev tensor move assignment
    Tensor host_d_copy = Tensor(host_c_copy.owned_storage().value(), single_tile_shape, DataType::BFLOAT16, Layout::TILE);
    host_d_copy = std::move(dev_a_copy);
    pass &= (host_d_copy.storage_type() == StorageType::DEVICE);
    auto host_d_copy_on_host = host_d_copy.to(host);
    auto host_d_copy_data = owned_buffer::get_as<bfloat16>(host_d_copy_on_host);
    pass &= host_d_copy_data == bfloat_data_view;

    // dev tensor updated with host tensor copy assignment
    auto random_tensor_four = tt::numpy::random::uniform(bfloat16(-1.0f), bfloat16(1.0f), single_tile_shape);
    auto bfloat_data_four = random_tensor_four.owned_storage().value().buffer;
    Tensor host_e = Tensor(random_tensor_four.owned_storage().value(), single_tile_shape, DataType::BFLOAT16, Layout::TILE);
    Tensor dev_e_copy = Tensor(host_c_copy.owned_storage().value(), single_tile_shape, DataType::BFLOAT16, Layout::TILE).to(device);
    dev_e_copy = std::move(host_e);
    pass &= (dev_e_copy.storage_type() == StorageType::OWNED);
    auto dev_e_copy_data = owned_buffer::get_as<bfloat16>(dev_e_copy);
    pass &= dev_e_copy_data == owned_buffer::get_as<bfloat16>(bfloat_data_four);

    // dev tensor updated with dev tensor copy assignment
    auto random_tensor_five = tt::numpy::random::uniform(bfloat16(-1.0f), bfloat16(1.0f), single_tile_shape);
    auto bfloat_data_five = random_tensor_four.owned_storage().value().buffer;
    Tensor dev_b = Tensor(random_tensor_four.owned_storage().value(), single_tile_shape, DataType::BFLOAT16, Layout::TILE).to(device);
    Tensor dev_b_copy = Tensor(dev_e_copy.owned_storage().value(), single_tile_shape, DataType::BFLOAT16, Layout::TILE).to(device);
    dev_b_copy = std::move(dev_b);
    pass &= (dev_b_copy.storage_type() == StorageType::DEVICE);
    auto dev_b_copy_on_host = dev_b_copy.to(host);
    auto dev_b_copy_data = owned_buffer::get_as<bfloat16>(dev_b_copy_on_host);
    pass &= dev_b_copy_data == owned_buffer::get_as<bfloat16>(bfloat_data_five);

    return pass;
}


int main(int argc, char **argv) {
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Initial Runtime Args Parse
        ////////////////////////////////////////////////////////////////////////////
        std::vector<std::string> input_args(argv, argv + argc);
        string arch_name = "";
        try {
            std::tie(arch_name, input_args) =
                test_args::get_command_option_and_remaining_args(input_args, "--arch", "grayskull");
        } catch (const std::exception& e) {
            log_fatal(tt::LogTest, "Command line arguments found exception", e.what());
        }
        const tt::ARCH arch = tt::get_arch_from_string(arch_name);
        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int pci_express_slot = 0;
        tt_metal::Device *device =
            tt_metal::CreateDevice(arch, pci_express_slot);
        pass &= tt_metal::InitializeDevice(device);
        tt_metal::Host *host = tt_metal::GetHost();

        pass &= test_tensor_copy_semantics(device, host);

        pass &= test_tensor_move_semantics(device, host);

        pass &= tt_metal::CloseDevice(device);

    } catch (const std::exception &e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        log_fatal(LogTest, "Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
