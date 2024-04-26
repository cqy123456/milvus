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
namespace milvus::storage {
/**
 * @brief keeping all information of tmp files
 */
struct MmapState {
 public:
    MmapState(const uint64_t disk_limit,
              const uint64_t fix_file_size,
              const std::string file_prefix)
        : max_disk_limit(disk_limit),
          mmap_file_prefix(file_prefix),
          fix_mmap_file_size(fix_file_size) {
        mmmap_file_counter.store(0);
    }
    std::string
    GetMmapFilePath() {
        auto file_id = mmmap_file_counter.fetch_add(1);
        return mmap_file_prefix + "/" + std::to_string(file_id);
    }
    uint64_t
    GetDiskLimit() {
        return max_disk_limit;
    }
    uint64_t
    GetFixFileSize() {
        return fix_mmap_file_size;
    }
    std::string
    GetFilePrefix() {
        return mmap_file_prefix;
    }

 private:
    uint64_t max_disk_limit;
    std::string mmap_file_prefix;
    std::atomic<uint64_t> mmmap_file_counter;
    uint64_t fix_mmap_file_size;
};
using MmapStatePtr = std::shared_ptr<MmapState>;

/**
 * @brief MmapEntry handle all memory mmaping in one tmp file
 */
struct MmapEntry {
 public:
    enum class EntryType {
        NORMAL = 0,
        LARGE = 1,
    };
    // create a new mmap file and init some objs
    MmapEntry(const std::string& file_name,
              const uint64_t file_size,
              EntryType type = EntryType::NORMAL);
    ~MmapEntry();
    void
    Clear() {
        offset_.store(0);
    }
    void*
    Allocate(const uint64_t size, ErrorCode& error_code);
    void
    Reset() {
        return offset_.store(0);
    }
    EntryType
    GetType() {
        return entry_type_;
    }
    uint64_t
    GetCapacity() {
        return file_size_;
    }

 private:
    const std::string file_name_;
    const uint64_t file_size_;
    char* addr_ = nullptr;
    std::atomic<uint64_t> offset_ = 0;
    const EntryType entry_type_;
};
using MmapEntryPtr = std::shared_ptr<MmapEntry>;

/*
* @brief keeping free tmp files in a queue 
*/
class FixedSizeMmapEntryQueue {
 public:
    FixedSizeMmapEntryQueue(const MmapStatePtr& entry_meta_ptr)
        : mmap_entry_meta_ptr_(entry_meta_ptr) {
        free_disk_size_.store(0);
    }
    MmapEntryPtr
    Pop();
    void
    Push(MmapEntryPtr&& entry);
    void
    Fit(const uint64_t size);
    void
    Clear();
    uint64_t
    GetSize() {
        return free_disk_size_.load();
    }

 private:
    //std::shared_mutex entry_mtx_;
    MmapStatePtr mmap_entry_meta_ptr_;
    std::queue<MmapEntryPtr> queuing_entries_;
    std::atomic<uint64_t> free_disk_size_;  // bytes
};

/**
 * @brief GrowingMmapFileManager
 */
class GrowingMmapChunkManager {
 public:
    GrowingMmapChunkManager(const std::string prefix,
                            const uint64_t max_limit,
                            const uint64_t file_size);
    ~GrowingMmapChunkManager();
    void
    Register(const uint64_t key);
    void
    UnRegister(const uint64_t key);
    inline bool
    HasKey(const uint64_t key);
    void*
    Allocate(const uint64_t key, const uint64_t size, ErrorCode& error_code);
    uint64_t
    GetDiskUsage() {
        return used_disk_size_.load();
    }
    uint64_t
    GetDiskAllocSize() {
        return queuing_entries_.GetSize() + used_disk_size_.load();
    }

 private:
    MmapStatePtr mmap_entry_meta_ptr_ = nullptr;
    // keep two struct(queue_entries_ and activate_entries_) to maintain (idle and busy) entries;
    mutable std::shared_mutex activate_entries_mtx_;
    std::unordered_map<uint64_t, std::vector<MmapEntryPtr>> activate_entries_;
    FixedSizeMmapEntryQueue queuing_entries_;
    std::atomic<uint64_t> used_disk_size_;  // bytes
};
using GrowingMmapChunkManagerPtr = std::shared_ptr<GrowingMmapChunkManager>;
using MmapChunkDescriptor =
    std::shared_ptr<std::pair<std::uint64_t, GrowingMmapChunkManagerPtr>>;
}  // namespace milvus::storage