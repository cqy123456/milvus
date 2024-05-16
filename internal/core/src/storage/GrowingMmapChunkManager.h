// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
#include <queue>
#include <atomic>
#include <unordered_map>
#include <shared_mutex>
#include <memory>
#include <shared_mutex>
#include "common/EasyAssert.h"
#include "log/Log.h"
#include <optional>
#include "storage/LocalChunkManagerSingleton.h"
namespace milvus::storage {
/**
 * @brief MmapBlock is a basic unit of MmapChunkManager. It handle all memory mmaping in one tmp file.
 * static function(TotalBlocksSize) is used to get total files size of chunk mmap.
 */
struct MmapBlock {
 public:
    enum class BlockType {
        Fixed = 0,
        Variable = 1,
    };
    MmapBlock(const std::string& file_name,
              const uint64_t file_size,
              BlockType type = BlockType::Fixed);
    ~MmapBlock();
    void
    Init();
    void
    Close();
    void*
    Get(const uint64_t size, ErrorCode& error_code);
    void
    Reset() {
        offset_.store(0);
    }
    BlockType
    GetType() {
        return block_type_;
    }
    uint64_t
    GetCapacity() {
        return file_size_;
    }
    static void
    ClearAllocSize() {
        allocated_size_.store(0);
    }
    static uint64_t
    TotalBlocksSize() {
        return allocated_size_.load();
    }

 private:
    const std::string file_name_;
    const uint64_t file_size_;
    char* addr_ = nullptr;
    std::atomic<uint64_t> offset_ = 0;
    const BlockType block_type_;
    std::atomic<bool> is_valid_ = false;
    static inline std::atomic<uint64_t> allocated_size_ =
        0;  //keeping the total size used in
};
using MmapBlockPtr = std::unique_ptr<MmapBlock>;

/**
 * @brief MmapBlocksHandler is used to handle the creation and destruction of mmap blocks
 */
class MmapBlocksHandler {
 public:
    MmapBlocksHandler(const uint64_t disk_limit,
                      const uint64_t fix_file_size,
                      const std::string file_prefix)
        : max_disk_limit_(disk_limit),
          mmap_file_prefix_(file_prefix),
          fix_mmap_file_size_(fix_file_size) {
        mmmap_file_counter_.store(0);
        MmapBlock::ClearAllocSize();
    }
    ~MmapBlocksHandler() {
        ClearCache();
    }
    uint64_t
    GetDiskLimit() {
        return max_disk_limit_;
    }
    uint64_t
    GetFixFileSize() {
        return fix_mmap_file_size_;
    }
    uint64_t
    Capacity() {
        return MmapBlock::TotalBlocksSize();
    }
    uint64_t
    Size() {
        return Capacity() - fix_size_blocks_cache_.size() * fix_mmap_file_size_;
    }
    MmapBlockPtr
    AllocateFixSizeBlock();
    MmapBlockPtr
    AllocateLargeBlock(const uint64_t size);
    void
    Deallocate(MmapBlockPtr&& block);

 private:
    std::string
    GetFilePrefix() {
        return mmap_file_prefix_;
    }
    std::string
    GetMmapFilePath() {
        auto file_id = mmmap_file_counter_.fetch_add(1);
        return mmap_file_prefix_ + "/" + std::to_string(file_id);
    }
    void
    ClearCache();
    void
    FitCache(const uint64_t size);

 private:
    uint64_t max_disk_limit_;
    std::string mmap_file_prefix_;
    std::atomic<uint64_t> mmmap_file_counter_;
    uint64_t fix_mmap_file_size_;
    std::queue<MmapBlockPtr> fix_size_blocks_cache_;
    const float cache_threshold = 0.25;
};

/**
 * @brief GrowingMmapFileManager(singleton)
 * GrowingMmapFileManager manages the memory-mapping space of all growing segments;
 * GrowingMmapFileManager uses blocks_table_ to record the relationship of growing segment and the mapp space it uses.
 * The basic space unit of MmapChunkManager is MmapBlock, and is managed by MmapBlocksHandler.
 */
class MmapChunkManager {
 public:
    MmapChunkManager(const MmapChunkManager&)=delete;
    MmapChunkManager& operator=(const MmapChunkManager&)=delete;
    static MmapChunkManager& get_instance(){
        static MmapChunkManager instance;
        return instance;
    }
    ~MmapChunkManager();
    void
    Register(const uint64_t key);
    void
    UnRegister(const uint64_t key);
    inline bool
    HasKey(const uint64_t key);
    void*
    Allocate(const uint64_t key, const uint64_t size);
    uint64_t
    GetDiskAllocSize() {
        std::shared_lock<std::shared_mutex> lck(mtx_);
        return blocks_handler_->Capacity();
    }
    uint64_t
    GetDiskUsage() {
        std::shared_lock<std::shared_mutex> lck(mtx_);
        return blocks_handler_->Size();
    }
    void
    Init(std::string root_path,
                         const uint64_t disk_limit,
                         const uint64_t file_size) {
        blocks_handler_= std::make_unique<MmapBlocksHandler>(disk_limit, file_size, root_path);
        mmap_file_prefix_ = root_path;
        auto cm =
            storage::LocalChunkManagerSingleton::GetInstance().GetChunkManager();
        AssertInfo(cm != nullptr,
                "Fail to get LocalChunkManager, LocalChunkManagerPtr is null");
        if (cm->Exist(root_path)) {
            cm->RemoveDir(root_path);
        }
        cm->CreateDir(root_path);

        LOG_INFO(
            "Init MappChunkManage with: Path {}, MaxDiskSize {} MB, "
            "FixedFileSize {} MB.",
            root_path,
            disk_limit / (1024 * 1024),
            file_size / (1024 * 1024));
    }

 private:
    MmapChunkManager(){}
    mutable std::shared_mutex mtx_;
    std::unordered_map<uint64_t, std::vector<MmapBlockPtr>> blocks_table_;
    std::unique_ptr<MmapBlocksHandler> blocks_handler_ = nullptr;
    std::string mmap_file_prefix_;
   // static std::shared_ptr<MmapChunkManager> chunk_manager_;
};
using GrowingMmapChunkManagerPtr = std::shared_ptr<MmapChunkManager>;
using MmapChunkKey = std::optional<std::uint64_t>;

}  // namespace milvus::storage