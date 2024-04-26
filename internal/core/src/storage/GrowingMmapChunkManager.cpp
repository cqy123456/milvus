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

#include "storage/GrowingMmapChunkManager.h"
#include "storage/LocalChunkManagerSingleton.h"
#include <fstream>
#include <sys/mman.h>
#include <unistd.h>
#include "stdio.h"
#include <fcntl.h>
#include "log/Log.h"

namespace milvus::storage {
namespace {
static constexpr int kMmapDefaultProt = PROT_WRITE | PROT_READ;
static constexpr int kMmapDefaultFlags = MAP_SHARED;
};  // namespace

MappEntry::MappEntry(const std::string& file_name,
                     const uint64_t file_size,
                     EntryType type)
    : file_name_(file_name), file_size_(file_size), entry_type_(type) {
    // create tmp file
    int fd = open(file_name_.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
    if (fd == -1) {
        LOG_ERROR("Failed to open mmap tmp file");
        return;
    }
    // append file size to 'file_size'
    if (lseek(fd, file_size - 1, SEEK_SET) == -1) {
        LOG_ERROR("Failed to seek mmap tmp file");
        return;
    }
    if (write(fd, "", 1) == -1) {
        LOG_ERROR("Failed to write mmap tmp file");
        return;
    }
    // memory mapping 
    addr_ = static_cast<char*>(mmap(nullptr, file_size, kMmapDefaultProt, kMmapDefaultFlags, fd, 0));
    if (addr_ == MAP_FAILED) {
        LOG_ERROR("Failed to mmap");
        return;
    }
    offset_.store(0);
}

MappEntry::~MappEntry() {
    if (addr_ != nullptr) {
        munmap(addr_, file_size_);
    }
    if (std::ifstream(file_name_.c_str()).good()) {
        std::remove(file_name_.c_str());
    }
}

void*
MappEntry::Allocate(const uint64_t size, ErrorCode& error_code) {
    if (file_size_ - offset_.load() < size) {
        error_code = ErrorCode::MemAllocateFailed;
        return nullptr;
    } else {
        error_code = ErrorCode::Success;
        return (void*)(addr_ + offset_.fetch_add(size));
    }
}

MappEntryPtr
FixedSizeMmapEntryQueue::Pop() {
    if (queuing_entries_.empty()) {
        return std::make_unique<MappEntry>(
            mmap_entry_meta_ptr_->GetMmapFilePath(),
            mmap_entry_meta_ptr_->GetFixFileSize(),
            MappEntry::EntryType::NORMAL);
    } else {
        std::lock_guard<std::shared_mutex> lock(entry_mtx_);
        auto entry = queuing_entries_.back();
        free_disk_size_.fetch_sub(entry->GetCapacity());
        queuing_entries_.pop();
        return entry;
    }
}

void
FixedSizeMmapEntryQueue::Push(MappEntryPtr&& entry) {
    free_disk_size_.fetch_add(entry->GetCapacity());
    {
        std::lock_guard<std::shared_mutex> lock(entry_mtx_);
        queuing_entries_.push(std::move(entry));
    }
    if (free_disk_size_ >= mmap_entry_meta_ptr_->GetDiskLimit() / 2) {
        Fit(mmap_entry_meta_ptr_->GetDiskLimit() / 2);
    }
}

void
FixedSizeMmapEntryQueue::Fit(const uint64_t size) {
    std::lock_guard<std::shared_mutex> lock(entry_mtx_);
    while (free_disk_size_.load() < size) {
        free_disk_size_.fetch_sub(mmap_entry_meta_ptr_->GetFixFileSize());
        queuing_entries_.pop();
    }
}

void
FixedSizeMmapEntryQueue::Clear() {
    std::lock_guard<std::shared_mutex> lock(entry_mtx_);
    while (!queuing_entries_.empty()) {
        queuing_entries_.pop();
    }
    free_disk_size_.store(0);
}

GrowingMmapChunkManager::GrowingMmapChunkManager(const std::string prefix,
                                                 const uint64_t max_limit,
                                                 const uint64_t file_size):
    mmap_entry_meta_ptr_(std::make_shared<MappState>(max_limit, file_size, prefix)),
    queuing_entries_(FixedSizeMmapEntryQueue(mmap_entry_meta_ptr_)) {
    auto cm =
        storage::LocalChunkManagerSingleton::GetInstance().GetChunkManager();
    AssertInfo(cm != nullptr,
               "Fail to get LocalChunkManager, LocalChunkManagerSPtr is null");

    if (cm->Exist(prefix)) {
        cm->RemoveDir(prefix);
    }
    cm->CreateDir(prefix);
    used_disk_size_.store(0);
}

GrowingMmapChunkManager::~GrowingMmapChunkManager() {
    auto prefix = mmap_entry_meta_ptr_->GetFilePrefix();
    auto cm =
        storage::LocalChunkManagerSingleton::GetInstance().GetChunkManager();
    if (cm->Exist(prefix)) {
        cm->RemoveDir(prefix);
    }
}

void
GrowingMmapChunkManager::Register(const uint64_t key) {
    if (HasKey(key)) {
        LOG_WARN("key has exist in growing mmap manager");
        return;
    }
    std::unique_lock<std::shared_mutex> lock(activate_entries_mtx_);
    activate_entries_.emplace(key, std::vector<MappEntryPtr>());
    return;
}

void
GrowingMmapChunkManager::UnRegister(const uint64_t key) {
    std::unique_lock<std::shared_mutex> lock(activate_entries_mtx_);
    if (activate_entries_.find(key) != activate_entries_.end()) {
        auto& entries = activate_entries_[key];
        for (auto i = 0; i < entries.size(); i++) {
            if (entries[i]->GetType() == MappEntry::EntryType::NORMAL) {
                entries[i]->Clear();
                queuing_entries_.Push(std::move(entries[i]));
            } else {
                entries[i] = nullptr;
            }
        }
        activate_entries_.erase(key);
    }
}

bool
GrowingMmapChunkManager::HasKey(const uint64_t key) {
    std::shared_lock<std::shared_mutex> lock(activate_entries_mtx_);
    return (activate_entries_.find(key) != activate_entries_.end());
}

void*
GrowingMmapChunkManager::Allocate(const uint64_t key,
                                  const uint64_t size,
                                  ErrorCode& error_code) {
    if (!HasKey(key)) {
        LOG_INFO("fail to alloc memory in mmap way, allocate key not exist.");
        error_code = ErrorCode::MemAllocateFailed;
        return nullptr;
    }
    // find a place to fix in
    {
        std::shared_lock<std::shared_mutex> lock(activate_entries_mtx_);
        for (auto entry_id = 0; entry_id < activate_entries_[key].size();
             entry_id++) {
            auto addr =
                activate_entries_[key][entry_id]->Allocate(size, error_code);
            if (error_code == ErrorCode::Success) {
                return addr;
            }
        }
    }
    // not enough disk space to allocate
    if (size + GetDiskUsage() > mmap_entry_meta_ptr_->GetDiskLimit()) {
        error_code = ErrorCode::MemAllocateFailed;
        return nullptr;
    }
    // get a new file to fit in
    MappEntryPtr entry = nullptr;
    void* addr = nullptr;
    if (size > mmap_entry_meta_ptr_->GetFixFileSize()) {
        // clear queuing_entries_ to create a new file
        if (size + GetDiskAllocSize() > mmap_entry_meta_ptr_->GetDiskLimit()) {
            queuing_entries_.Clear();
        }
        entry =
            std::make_unique<MappEntry>(mmap_entry_meta_ptr_->GetMmapFilePath(),
                                        size,
                                        MappEntry::EntryType::LARGE);
    } else {
        entry = queuing_entries_.Pop();
    }

    if (entry != nullptr) {
        std::unique_lock<std::shared_mutex> lock(activate_entries_mtx_);
        addr = entry->Allocate(size, error_code);
        activate_entries_[key].emplace_back(std::move(entry));
    } else {
        LOG_ERROR("fail to get a mmap entry");
        error_code = ErrorCode::MemAllocateFailed;
        return nullptr;
    }
    return addr;
}
}  // namespace milvus::storage