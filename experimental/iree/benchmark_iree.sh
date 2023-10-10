#!/bin/bash

# This is a temporary hack to run IREE benchmarks on pixel-6-pro since
# it's currently not working in the IREE repo.

ROOT_DIR=/tmp/iree-benchmarks
TD="$(cd $(dirname $0) && pwd)"

rm -rf "${ROOT_DIR}"
mkdir "${ROOT_DIR}"
pushd "${ROOT_DIR}"

# Download benchmark tool.
gsutil cp "gs://iree-github-actions-presubmit-artifacts/6464567954/1/benchmark-tools/android-armv8.2-a-benchmark-tools.tar" .
tar -xf "android-armv8.2-a-benchmark-tools.tar"
adb push "android-armv8.2-a-benchmark-tools-dir/build/tools/iree-benchmark-module" "/data/local/tmp"
adb shell "chmod +x /data/local/tmp/iree-benchmark-module"

# Download vmfb's.


# Setup environment.
adb push "${TD}/set_android_scaling_governor.sh" "/data/local/tmp"
adb shell "chmod +x /data/local/tmp/set_android_scaling_governor.sh"
adb shell "su root sh /data/local/tmp/set_android_scaling_governor.sh performance"

# Benchmark.
ITERATIONS=10
gsutil cp "gs://iree-github-actions-presubmit-artifacts/6464567954/1/e2e-test-artifacts/iree_module_BertLarge_Fp32_Batch1_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_data-tiling_ukernel_/module.vmfb" "BertLarge_Batch1.vmfb"
adb push "BertLarge_Batch1.vmfb" "/data/local/tmp"
adb shell "taskset f0 /data/local/tmp/iree-benchmark-module --function=main --input=1x384xi32=0 --input=1x384xi32=0 --device_allocator=caching --task_topology_group_count=4 --device=local-task --module=/data/local/tmp/BertLarge_Batch1.vmfb --time_unit=ns --benchmark_format=json --benchmark_out_format=json --print_statistics=true --benchmark_repetitions=${ITERATIONS}"
adb shell "rm /data/local/tmp/BertLarge_Batch1.vmfb"
rm "BertLarge_Batch1.vmfb"

gsutil cp "gs://iree-github-actions-presubmit-artifacts/6464567954/1/e2e-test-artifacts/iree_module_BertLarge_Fp32_Batch16_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_data-tiling_ukernel_/module.vmfb" "BertLarge_Batch16.vmfb"
adb push "BertLarge_Batch16.vmfb" "/data/local/tmp"
adb shell "taskset f0 /data/local/tmp/iree-benchmark-module --function=main --input=16x384xi32=0 --input=16x384xi32=0 --device_allocator=caching --task_topology_group_count=4 --device=local-task --module=/data/local/tmp/BertLarge_Batch16.vmfb --time_unit=ns --benchmark_format=json --benchmark_out_format=json --print_statistics=true --benchmark_repetitions=${ITERATIONS}"
adb shell "rm /data/local/tmp/BertLarge_Batch16.vmfb"
rm "BertLarge_Batch16.vmfb"

gsutil cp "gs://iree-github-actions-presubmit-artifacts/6464567954/1/e2e-test-artifacts/iree_module_BertLarge_Fp32_Batch24_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_data-tiling_ukernel_/module.vmfb" "BertLarge_Batch24.vmfb"
adb push "BertLarge_Batch24.vmfb" "/data/local/tmp"
adb shell "taskset f0 /data/local/tmp/iree-benchmark-module --function=main --input=24x384xi32=0 --input=24x384xi32=0 --device_allocator=caching --task_topology_group_count=4 --device=local-task --module=/data/local/tmp/BertLarge_Batch24.vmfb --time_unit=ns --benchmark_format=json --benchmark_out_format=json --print_statistics=true --benchmark_repetitions=${ITERATIONS}"
adb shell "rm /data/local/tmp/BertLarge_Batch24.vmfb"
rm "BertLarge_Batch24.vmfb"

gsutil cp "gs://iree-github-actions-presubmit-artifacts/6464567954/1/e2e-test-artifacts/iree_module_BertLarge_Fp32_Batch32_tflite___armv8.2-a-generic-linux_android29-llvm_cpu__experimental-flags_data-tiling_ukernel_/module.vmfb" "BertLarge_Batch32.vmfb"
adb push "BertLarge_Batch32.vmfb" "/data/local/tmp"
adb shell "taskset f0 /data/local/tmp/iree-benchmark-module --function=main --input=32x384xi32=0 --input=32x384xi32=0 --device_allocator=caching --task_topology_group_count=4 --device=local-task --module=/data/local/tmp/BertLarge_Batch32.vmfb --time_unit=ns --benchmark_format=json --benchmark_out_format=json --print_statistics=true --benchmark_repetitions=${ITERATIONS}"
adb shell "rm /data/local/tmp/BertLarge_Batch32.vmfb"
rm "BertLarge_Batch32.vmfb"

adb shell "rm -rf /data/local/tmp/*"

popd
rm -rf "${ROOT_DIR}"





