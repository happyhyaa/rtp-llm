#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/test/MockAsyncContext.h"

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <utility>

namespace rtp_llm {
namespace test {

namespace {

class BlockingAsyncContext final: public AsyncContext {
public:
    void setDone(bool done) {
        std::vector<DoneCallback> callbacks;
        {
            std::lock_guard<std::mutex> lk(mu_);
            done_ = done;
            if (done_) {
                callbacks.swap(callbacks_);
            }
        }
        cv_.notify_all();
        for (auto& callback : callbacks) {
            callback(success_.load(std::memory_order_relaxed) ?
                         ErrorInfo::OkStatus() :
                         ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "blocking context failed"));
        }
    }

    void setSuccess(bool success) {
        success_.store(success, std::memory_order_relaxed);
    }

    void waitDone() override {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return done_; });
    }

    void onDone(DoneCallback callback) override {
        if (!callback) {
            return;
        }
        bool run_now = false;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (done_) {
                run_now = true;
            } else {
                callbacks_.push_back(std::move(callback));
            }
        }
        if (run_now) {
            callback(success() ? ErrorInfo::OkStatus() :
                                 ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "blocking context failed"));
        }
    }

    bool done() const override {
        std::lock_guard<std::mutex> lk(mu_);
        return done_;
    }

    bool success() const override {
        return success_.load(std::memory_order_relaxed);
    }

private:
    mutable std::mutex              mu_;
    mutable std::condition_variable cv_;
    bool                            done_{false};
    std::atomic<bool>               success_{true};
    std::vector<DoneCallback>       callbacks_;
};

}  // namespace

TEST(AsyncContextTest, CompletedAsyncContext_SuccessStatus) {
    CompletedAsyncContext context(ErrorInfo::OkStatus());
    EXPECT_TRUE(context.done());
    EXPECT_TRUE(context.success());
    EXPECT_TRUE(context.errorInfo().ok());

    context.waitDone();
    EXPECT_TRUE(context.done());
    EXPECT_TRUE(context.success());
    EXPECT_TRUE(context.errorInfo().ok());
}

TEST(AsyncContextTest, CompletedAsyncContext_FailureStatusPreservesErrorInfo) {
    CompletedAsyncContext context(ErrorInfo(ErrorCode::INVALID_PARAMS, "invalid block transfer request"));
    EXPECT_TRUE(context.done());
    EXPECT_FALSE(context.success());
    EXPECT_FALSE(context.errorInfo().ok());
    EXPECT_EQ(context.errorInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(context.errorInfo().ToString(), "invalid block transfer request");

    context.waitDone();
    EXPECT_TRUE(context.done());
    EXPECT_FALSE(context.success());
    EXPECT_EQ(context.errorInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(context.errorInfo().ToString(), "invalid block transfer request");
}

TEST(AsyncContextTest, CompletedAsyncContext_OnDoneRunsImmediately) {
    CompletedAsyncContext context(ErrorInfo(ErrorCode::INVALID_PARAMS, "already done"));
    size_t                callback_count = 0;

    context.onDone([&](ErrorInfo error) {
        EXPECT_EQ(error.code(), ErrorCode::INVALID_PARAMS);
        ++callback_count;
    });

    EXPECT_EQ(callback_count, 1u);
}

TEST(AsyncContextTest, FusedAsyncContext_DoneTrue_WhenEmptyOrAllDoneOrNull) {
    auto c1 = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    auto c2 = std::shared_ptr<AsyncContext>{nullptr};
    ON_CALL(*c1, done()).WillByDefault(testing::Return(true));

    FusedAsyncContext fused_empty({});
    EXPECT_TRUE(fused_empty.done());

    FusedAsyncContext fused({c1, c2});
    EXPECT_TRUE(fused.done());
}

TEST(AsyncContextTest, FusedAsyncContext_DoneFalse_WhenAnyNotDone) {
    auto done_ctx     = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    auto not_done_ctx = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*done_ctx, done()).WillByDefault(testing::Return(true));
    ON_CALL(*not_done_ctx, done()).WillByDefault(testing::Return(false));

    FusedAsyncContext fused({done_ctx, not_done_ctx});
    EXPECT_FALSE(fused.done());
}

TEST(AsyncContextTest, FusedAsyncContext_SuccessTrue_WhenAllSuccessOrNull) {
    auto ok = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*ok, success()).WillByDefault(testing::Return(true));

    FusedAsyncContext fused({ok, nullptr});
    EXPECT_TRUE(fused.success());
}

TEST(AsyncContextTest, FusedAsyncContext_SuccessFalse_WhenAnyFail) {
    auto ok  = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    auto bad = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*ok, success()).WillByDefault(testing::Return(true));
    ON_CALL(*bad, success()).WillByDefault(testing::Return(false));

    FusedAsyncContext fused({ok, bad});
    EXPECT_FALSE(fused.success());
}

TEST(AsyncContextTest, FusedAsyncContext_OnDoneWaitsForAllChildrenAndKeepsFirstError) {
    auto first  = std::make_shared<BlockingAsyncContext>();
    auto second = std::make_shared<BlockingAsyncContext>();
    first->setSuccess(false);
    FusedAsyncContext fused({first, second});

    size_t    callback_count = 0;
    ErrorCode result_code    = ErrorCode::NONE_ERROR;
    fused.onDone([&](ErrorInfo error) {
        result_code = error.code();
        ++callback_count;
    });

    first->setDone(true);
    EXPECT_EQ(callback_count, 0u);
    second->setDone(true);
    EXPECT_EQ(callback_count, 1u);
    EXPECT_EQ(result_code, ErrorCode::EXECUTION_EXCEPTION);
}

TEST(AsyncContextTest, FusedAsyncReadContext_DoneTrue_WhenMatchContextNull) {
    auto resource = std::shared_ptr<KVCacheResource>{};  // not used by logic
    auto ctx      = std::make_shared<FusedAsyncReadContext>(nullptr, resource);
    EXPECT_TRUE(ctx->done());
    EXPECT_FALSE(ctx->success());  // fused_match_context_ is null => success() must be false
}

TEST(AsyncContextTest, FusedAsyncReadContext_DoneFalse_WhenMatchNotDone) {
    auto match_child = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*match_child, done()).WillByDefault(testing::Return(false));
    ON_CALL(*match_child, success()).WillByDefault(testing::Return(true));

    auto match = std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{match_child});
    auto ctx   = std::make_shared<FusedAsyncReadContext>(match, std::shared_ptr<KVCacheResource>{});

    EXPECT_FALSE(ctx->done());
    EXPECT_FALSE(ctx->success());
}

TEST(AsyncContextTest, FusedAsyncReadContext_DoneTrue_WhenMatchDoneButFailed) {
    auto match_child = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*match_child, done()).WillByDefault(testing::Return(true));
    ON_CALL(*match_child, success()).WillByDefault(testing::Return(false));

    auto match = std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{match_child});
    auto ctx   = std::make_shared<FusedAsyncReadContext>(match, std::shared_ptr<KVCacheResource>{});

    EXPECT_TRUE(ctx->done());
    EXPECT_FALSE(ctx->success());
}

TEST(AsyncContextTest, FusedAsyncReadContext_DoneFalse_WhenMatchSuccessButReadNotSet) {
    auto match_child = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*match_child, done()).WillByDefault(testing::Return(true));
    ON_CALL(*match_child, success()).WillByDefault(testing::Return(true));

    auto match = std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{match_child});
    auto ctx   = std::make_shared<FusedAsyncReadContext>(match, std::shared_ptr<KVCacheResource>{});

    EXPECT_FALSE(ctx->done());
    EXPECT_FALSE(ctx->success());
}

TEST(AsyncContextTest, FusedAsyncReadContext_SuccessDependsOnReadContext) {
    auto match_child = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*match_child, done()).WillByDefault(testing::Return(true));
    ON_CALL(*match_child, success()).WillByDefault(testing::Return(true));
    auto match = std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{match_child});

    auto read_ok = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*read_ok, done()).WillByDefault(testing::Return(true));
    ON_CALL(*read_ok, success()).WillByDefault(testing::Return(true));
    auto read_fused = std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{read_ok});

    auto ctx = std::make_shared<FusedAsyncReadContext>(match, std::shared_ptr<KVCacheResource>{});
    ctx->setFusedReadContext(read_fused);

    EXPECT_TRUE(ctx->done());
    EXPECT_TRUE(ctx->success());

    // Now make read context fail while done.
    auto read_bad = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*read_bad, done()).WillByDefault(testing::Return(true));
    ON_CALL(*read_bad, success()).WillByDefault(testing::Return(false));
    ctx->setFusedReadContext(std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{read_bad}));

    EXPECT_TRUE(ctx->done());
    EXPECT_FALSE(ctx->success());
}

TEST(AsyncContextTest, FusedAsyncReadContext_WaitDone_WaitsForLateReadContext) {
    // match succeeds quickly
    auto match_child = std::make_shared<BlockingAsyncContext>();
    match_child->setSuccess(true);
    match_child->setDone(true);
    auto match = std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{match_child});

    auto ctx  = std::make_shared<FusedAsyncReadContext>(match, std::shared_ptr<KVCacheResource>{});

    std::atomic<bool> returned{false};
    std::thread       waiter([&] {
        ctx->waitDone();
        returned.store(true, std::memory_order_release);
    });

    // Should not return yet because read context has not been set.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_FALSE(returned.load(std::memory_order_acquire));

    // Now set a read context that is not done yet.
    auto read_child = std::make_shared<BlockingAsyncContext>();
    read_child->setSuccess(true);
    read_child->setDone(false);
    auto read = std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{read_child});
    ctx->setFusedReadContext(read);
    // The producer calls `notifyDone()` whenever the staged read state changes.
    ctx->notifyDone();

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_FALSE(returned.load(std::memory_order_acquire));

    // Complete read stage and ensure waiter returns.
    read_child->setDone(true);
    ctx->notifyDone();
    waiter.join();
    EXPECT_TRUE(returned.load(std::memory_order_acquire));
}

TEST(AsyncContextTest, FusedAsyncReadContext_WaitDone_ReturnsWhenLateReadContextSetNull) {
    // match succeeds quickly
    auto match_child = std::make_shared<BlockingAsyncContext>();
    match_child->setSuccess(true);
    match_child->setDone(true);
    auto match = std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{match_child});

    auto ctx  = std::make_shared<FusedAsyncReadContext>(match, std::shared_ptr<KVCacheResource>{});

    std::atomic<bool> returned{false};
    std::thread       waiter([&] {
        ctx->waitDone();
        returned.store(true, std::memory_order_release);
    });

    // Should not return yet because read context has not been set.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_FALSE(returned.load(std::memory_order_acquire));

    // Set read context explicitly to nullptr (skip read stage) => waitDone should return.
    ctx->setFusedReadContext(nullptr);
    ctx->notifyDone();
    waiter.join();
    EXPECT_TRUE(returned.load(std::memory_order_acquire));
}

TEST(AsyncContextTest, FusedAsyncReadContext_WaitDone_ReturnsWhenMatchDoneButFailed) {
    // match fails quickly
    auto match_child = std::make_shared<BlockingAsyncContext>();
    match_child->setSuccess(false);
    match_child->setDone(true);
    auto match = std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{match_child});

    auto ctx  = std::make_shared<FusedAsyncReadContext>(match, std::shared_ptr<KVCacheResource>{});

    std::atomic<bool> returned{false};
    std::thread       waiter([&] {
        ctx->waitDone();
        returned.store(true, std::memory_order_release);
    });

    // Should return quickly even though read context is never set.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    ctx->notifyDone();
    waiter.join();
    EXPECT_TRUE(returned.load(std::memory_order_acquire));
}

TEST(AsyncContextTest, FusedAsyncReadContext_DoneTrue_WhenReadContextSetNullAfterMatchSuccess) {
    auto match_child = std::make_shared<testing::NiceMock<MockAsyncContext>>();
    ON_CALL(*match_child, done()).WillByDefault(testing::Return(true));
    ON_CALL(*match_child, success()).WillByDefault(testing::Return(true));
    auto match = std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{match_child});

    auto ctx  = std::make_shared<FusedAsyncReadContext>(match, std::shared_ptr<KVCacheResource>{});

    // Not set yet => not done
    EXPECT_FALSE(ctx->done());

    // Explicitly set to nullptr => skip read stage => done & success
    ctx->setFusedReadContext(nullptr);
    EXPECT_TRUE(ctx->done());
    EXPECT_TRUE(ctx->success());
}

}  // namespace test
}  // namespace rtp_llm
