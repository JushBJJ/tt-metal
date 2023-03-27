#include <cstdint>

#define ELTWISE_OP_CODE 0 // TODO(AP): temporary - refactor

#include "llk_3c.h"

namespace NAMESPACE {

#ifdef TRISC_MATH
#include <cstdint>
#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h"
#include "llk_math_eltwise_unary_datacopy.h"

void math_main()
{
    uint32_t per_core_num_blocks = get_compile_time_arg_val(0);
    uint32_t per_core_block_r_tiles = get_compile_time_arg_val(1);
    uint32_t per_core_block_c_tiles = get_compile_time_arg_val(2);

    llk_math_pack_sync_init<SyncHalf>();
    for (uint32_t block = 0; block < per_core_num_blocks; block++) {
        for (uint32_t r = 0; r < per_core_block_r_tiles; r++) {
            // Untilize
            llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE, false>();
            for (uint32_t c = 0; c < per_core_block_c_tiles; c++) {
                llk_math_wait_for_dest_available<SyncHalf>();
                llk_math_eltwise_unary_datacopy<A2D, BroadcastType::NONE, SyncHalf>(0);
                llk_math_dest_section_done<SyncHalf>();
            }

            llk_math_eltwise_binary_init<ELWADD, NONE>();
            for (uint32_t c = 0; c < per_core_block_c_tiles; c++) {
                llk_math_wait_for_dest_available<SyncHalf>();
                llk_math_eltwise_binary<ELWADD, NONE, SyncHalf, MATH_FIDELITY, false>(0);
                llk_math_dest_section_done<SyncHalf>();
            }
        }
    }
}
#endif

#ifdef TRISC_UNPACK
#include <cstdint>
#include "llk_unpack_common.h"
#include "llk_unpack_AB.h"
#include "llk_unpack_untilize.h"

void unpack_main()
{
uint32_t per_core_num_blocks = get_compile_time_arg_val(0);
uint32_t per_core_block_r_tiles = get_compile_time_arg_val(1);
uint32_t per_core_block_c_tiles = get_compile_time_arg_val(2);

llk_setup_operands();
llk_unpack_AB_hw_configure_disaggregated<BroadcastType::NONE>(0,1);
// llk_unpack_untilize_hw_configure_disaggregated(0);

// volatile uint32_t* mbox = reinterpret_cast<volatile uint32_t*>(l1_mem::address_map::TRISC0_DEBUG_BUFFER_BASE);

// llk_unpack_untilize_init(0);
for (uint32_t block = 0U; block < per_core_num_blocks; ++block) {
  for (uint32_t r = 0; r < per_core_block_r_tiles; r++) {
    llk_unpack_untilize_init(0);
    llk_wait_tiles(0, per_core_block_c_tiles);
    llk_unpack_untilize(0, per_core_block_c_tiles);
    llk_unpack_untilize_uninit(0);
    llk_pop_tiles(0, per_core_block_c_tiles);
    llk_pop_tiles(1, per_core_block_c_tiles);

    llk_unpack_AB_init<BroadcastType::NONE>();
    for (uint32_t c = 0; c < per_core_block_c_tiles; c++) {
        llk_wait_tiles(24, 1);
        llk_wait_tiles(1, 1);
        llk_unpack_AB(24, 1, 0, 0);
        llk_pop_tiles(24, 1);
        llk_pop_tiles(1, 1);
    }
  }
}
}
#endif


#ifdef TRISC_PACK
#include <cstdint>
#include "llk_pack_common.h"
#include "llk_pack.h"

void pack_main()
{
    uint32_t per_core_num_blocks = get_compile_time_arg_val(0);
    uint32_t per_core_block_r_tiles = get_compile_time_arg_val(1);
    uint32_t per_core_block_c_tiles = get_compile_time_arg_val(2);
    llk_pack_init();
    llk_pack_hw_configure_disaggregated<false>(16);
    llk_setup_outputs();
    llk_pack_dest_init<SyncHalf, DstTileFaceLayout::RowMajor, false>();
    volatile uint32_t* mbox = reinterpret_cast<volatile uint32_t*>(l1_mem::address_map::TRISC0_DEBUG_BUFFER_BASE);

    for (uint32_t block = 0; block < per_core_num_blocks; block++) {
        for (uint32_t r = 0; r < per_core_block_r_tiles; r++) {
            llk_wait_for_free_tiles<false,false,false>(24, per_core_block_c_tiles);
            for (uint32_t c = 0; c < per_core_block_c_tiles; c++) {
                llk_packer_wait_for_math_done();
                llk_pack<false, SyncHalf, false >(0,24);
                llk_pack_dest_section_done<SyncHalf>();
            }
            llk_push_tiles<false, false>(24, per_core_block_c_tiles);

            llk_wait_for_free_tiles<false,false,false>(16, per_core_block_c_tiles);
            for (uint32_t c = 0; c < per_core_block_c_tiles; c++) {
                llk_packer_wait_for_math_done();
                llk_pack<false, SyncHalf, false >(0,16);
                llk_pack_dest_section_done<SyncHalf>();
            }
            llk_push_tiles<false, false>(16, per_core_block_c_tiles);
        }
    }
}
#endif

} // NAMESPACE
