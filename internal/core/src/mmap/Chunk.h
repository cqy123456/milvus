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
#include "storage/GrowingMmapChunkManager.h"
namespace milvus {
template <typename Type>
struct Chunk {
    const int64_t size = 0;
    const Type* data = nullptr;
};
template <typename Type>
class ThreadSafeChunkVector {
 public:
    virtual void
    emplace_to_at_least(int64_t chunk_num, int64_t chunk_size) {
        if (chunk_num <= counter_) {
            return;
        }
        std::lock_guard lck(mutex_);
        while (vec_.size() < chunk_num) {
            vec_.emplace_back(chunk_size);
            ++counter_;
        }
    }

    virtual const Type*
    get_chunk_data(int64_t index) {
        AssertInfo(
            index < counter_,
            fmt::format(
                "index out of range, index={}, counter_={}", index, counter_));
        std::shared_lock lck(mutex_);
        return vec_[index].data();
    }

    virtual int64_t
    get_chunk_data_size(int64_t index) {
        AssertInfo(
            index < counter_,
            fmt::format(
                "index out of range, index={}, counter_={}", index, counter_));
        std::shared_lock lck(mutex_);
        return vec_[index].size();
    }

    virtual Chunk<Type>
    get_chunk(int64_t index) {
        return Chunk<Type>{vec_[index].size(), vec_[index].data()};
    }

    int64_t
    size() const {
        return counter_;
    }

    virtual void
    clear() {
        std::lock_guard lck(mutex_);
        counter_ = 0;
        vec_.clear();
    }

 protected:
    std::atomic<int64_t> counter_ = 0;
    mutable std::shared_mutex mutex_;

 private:
    using ChunkImpl = FixedVector<Type>;
    std::deque<ChunkImpl> vec_;
};
template <typename Type>
using ThreadSafeChunkVectorPtr = std::unique_ptr<ThreadSafeChunkVector<Type>>;

template <typename Type>
class MmapThreadSafeChunkVector : public ThreadSafeChunkVector<Type> {
 public:
    explicit MmapThreadSafeChunkVector(
        const storage::MmapChunkDescriptor& mmap_descriptor)
        : mmap_descriptor_(mmap_descriptor) {
    }
    void
    emplace_to_at_least(int64_t chunk_num, int64_t chunk_size) override {
        if (chunk_num <= this->counter_) {
            return;
        }
        auto& mmap_manager = mmap_descriptor_->second;
        milvus::ErrorCode err_code;
        auto data = (Type*)(mmap_manager->Allocate(
            mmap_descriptor_->first, sizeof(Type) * chunk_size, err_code));
        AssertInfo(data != nullptr,
                   "failed to create a mmapchunk: {}, map_size={}",
                   strerror(err_code),
                   chunk_num);

        std::lock_guard lck(this->mutex_);
        while (mmap_vec_.size() < chunk_num) {
            mmap_vec_.emplace_back(Chunk<Type>{chunk_size, data});
            ++this->counter_;
        }
    }

    const Type*
    get_chunk_data(int64_t index) override {
        AssertInfo(index < this->counter_,
                   fmt::format("index out of range, index={}, counter_={}",
                               index,
                               this->counter_));
        std::shared_lock lck(this->mutex_);
        return mmap_vec_[index].data;
    }

    int64_t
    get_chunk_data_size(int64_t index) override {
        AssertInfo(index < this->counter_,
                   fmt::format("index out of range, index={}, counter_={}",
                               index,
                               this->counter_));
        std::shared_lock lck(this->mutex_);
        return mmap_vec_[index].size;
    }

    void
    clear() override {
        // do not clear memory, it will handled by GrowingMmapChunkManager
        std::lock_guard lck(this->mutex_);
        this->counter_ = 0;
        mmap_vec_.clear();
    }

    Chunk<Type>
    get_chunk(int64_t index) override {
        return mmap_vec_[index];
    }

 private:
    std::deque<Chunk<Type>> mmap_vec_;
    storage::MmapChunkDescriptor mmap_descriptor_;
};

}  // namespace milvus