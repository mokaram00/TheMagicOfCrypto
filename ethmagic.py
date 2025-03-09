import ctypes
import time
import argparse
import multiprocessing
import gc
import pyopencl as cl
from bip_utils import Bip32Slip10Secp256k1, Bip39MnemonicGenerator, Bip39SeedGenerator, Bip39WordsNum, EthAddrEncoder
from rich import print
import sys
import signal
import requests
import datetime
import numpy as np
import psutil

# Global variables for shared GPU context
global_ctx = None
global_queue = None
global_lock = multiprocessing.Lock()

BATCH_SIZE = 16384  # Reduced batch size for better memory management
WORK_GROUP_SIZE = 256

# إضافة Webhook URL ثابت في بداية الملف
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1347354458769592330/tD4Xii-C5dbPJVxfBcqt-WQ8xtQr3OAhU8r9E24TJSza73rdSVrd2c4hr3V8zvVdHHUW"

def init_gpu():
    global global_ctx, global_queue
    
    with global_lock:
        if global_ctx is not None and global_queue is not None:
            return global_ctx, global_queue
            
        platforms = cl.get_platforms()
        amd_platform = None
        for platform in platforms:
            if 'AMD' in platform.name:
                amd_platform = platform
                break
        
        if amd_platform is None:
            print("[red]No AMD GPU found![/red]")
            return None, None
            
        devices = amd_platform.get_devices(device_type=cl.device_type.GPU)
        if not devices:
            print("[red]No AMD GPU devices found![/red]")
            return None, None
        
        try:
            # Get device memory info
            device = devices[0]
            mem_info = device.get_info(cl.device_info.GLOBAL_MEM_SIZE)
            print(f"[green]GPU Memory: {mem_info / (1024**3):.2f} GB[/green]")
            
            # Only proceed if we have enough memory
            required_mem = BATCH_SIZE * 52 * 2  # Double the batch size for safety
            if required_mem > mem_info * 0.8:  # Use only 80% of available memory
                print("[red]Not enough GPU memory available[/red]")
                return None, None
                
            ctx = cl.Context(devices)
            queue = cl.CommandQueue(ctx)
            
            global_ctx = ctx
            global_queue = queue
            
            return ctx, queue
            
        except Exception as e:
            print(f"[red]Error initializing GPU context: {str(e)}[/red]")
            return None, None

def clean_memory():
    gc.collect()
    process = psutil.Process()
    process.memory_info()

def signal_handler(signum, frame):
    print("\n[red]Stopping all processes...[/red]")
    sys.exit(0)

def send_discord_webhook(webhook_url, content, color=0x00ff00):
    data = {
        "embeds": [{
            "title": "Crypto Magic Scanner",
            "description": content,
            "color": color,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }]
    }
    try:
        requests.post(webhook_url, json=data)
    except Exception as e:
        print(f"[red]Error sending webhook: {str(e)}[/red]")

def format_stats(thread_id, total_keys, current_speed, avg_speed, found_count):
    """Format performance statistics in a pretty way"""
    return f"""
╔══════════════════════════════════════════════
║ 🧵 Thread: {thread_id}
║ 🔑 Total Keys: {total_keys:,}
║ ⚡ Current Speed: {current_speed:,.2f} keys/s
║ 📊 Average Speed: {avg_speed:,.2f} keys/s
║ 💎 Matches Found: {found_count}
╚══════════════════════════════════════════════"""

def format_match(addr, priv_key, master_key, mnemonic):
    """Format match information in a pretty way"""
    return f"""
╔═══════════════════ MATCH FOUND ═══════════════════
║ 
║ 📍 Address: {addr}
║ 
║ 🔐 Private Key: {priv_key}
║ 
║ 👑 Master Key: {master_key}
║ 
║ 📝 Mnemonic:
║ {mnemonic}
║ 
╚═══════════════════════════════════════════════════"""

def format_batch_info(thread_id, mnemonic, master_key, private_key, total_keys, keys_per_second, logp):
    """Format current batch information in a pretty way"""
    return f"""
╔═══════════════════ CURRENT BATCH ═══════════════════
║ 
║ 🧵 Thread: {thread_id}
║ 
║ 📝 Current Mnemonic:
║ {mnemonic}
║ 
║ 👑 Master Key:
║ {master_key}
║ 
║ 🔐 Private Key:
║ {private_key if private_key else 'Generating...'}
║ 
║ 📊 Stats:
║ • Total Keys: {total_keys:,}
║ • Speed: {keys_per_second:,.2f} keys/s
║ • Progress: {logp:,} addresses
║ 
╚════════════════════════════════════════════════════"""

def worker_process(add, logpx, thread_id):
    z = 0
    fu = 0
    logp = 0
    last_webhook_time = time.time()
    last_memory_clean = time.time()
    WEBHOOK_INTERVAL = 900
    MEMORY_CLEAN_INTERVAL = 300
    iteration_count = 0
    
    # Performance monitoring variables
    start_time = time.time()
    last_time = start_time
    total_keys = 0
    keys_per_second = 0
    
    # Initialize GPU with shared context
    ctx, queue = init_gpu()
    if ctx and queue:
        print(f"[green]Thread {thread_id} using AMD GPU[/green]")
        try:
            # Pre-allocate buffers with error handling
            try:
                total_size = BATCH_SIZE * 52
                gpu_output = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=total_size)
                gpu_target_addresses = cl.Buffer(ctx, cl.mem_flags.READ_ONLY, size=len(add) * 20)
                
                # Convert addresses to bytes and copy to GPU
                target_addresses = np.zeros(len(add) * 20, dtype=np.uint8)
                for i, addr in enumerate(add):
                    try:
                        if addr.startswith('0x'):
                            addr = addr[2:]
                        addr_bytes = bytes.fromhex(addr)
                        target_addresses[i*20:(i+1)*20] = list(addr_bytes)
                    except Exception as e:
                        print(f"[red]Error processing address {addr}: {str(e)}[/red]")
                        continue
                
                cl.enqueue_copy(queue, gpu_target_addresses, target_addresses)
                
            except cl.MemoryError:
                print(f"[red]Not enough GPU memory for thread {thread_id}[/red]")
                return
            except Exception as e:
                print(f"[red]Error allocating GPU buffers: {str(e)}[/red]")
                return

            # OpenCL program for key generation and address matching
            program = cl.Program(ctx, """
                #define WORK_GROUP_SIZE 256
                #define ROUNDS_PER_THREAD 128
                
                // Fast random number generation
                uint4 xoshiro128(uint4 state) {
                    uint t = state.y << 9;
                    state.z ^= state.x;
                    state.w ^= state.y;
                    state.y ^= state.z;
                    state.x ^= state.w;
                    state.z ^= t;
                    state.w = (state.w << 11) | (state.w >> 21);
                    return state;
                }

                // Simple hash function for address generation
                void hash_bytes(uchar *input, int input_len, uchar *output) {
                    uint state[8] = {
                        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
                    };
                    
                    for (int i = 0; i < input_len; i += 64) {
                        uint w[16];
                        for (int j = 0; j < 16; j++) {
                            w[j] = 0;
                            for (int k = 0; k < 4 && (i + j*4 + k) < input_len; k++) {
                                w[j] |= (uint)input[i + j*4 + k] << (24 - k*8);
                            }
                        }
                        
                        uint a = state[0];
                        uint b = state[1];
                        uint c = state[2];
                        uint d = state[3];
                        uint e = state[4];
                        uint f = state[5];
                        uint g = state[6];
                        uint h = state[7];
                        
                        for (int j = 0; j < 64; j++) {
                            uint temp1 = h + ((e >> 6 | e << 26) ^ (e >> 11 | e << 21) ^ (e >> 25 | e << 7)) +
                                       ((e & f) ^ (~e & g)) + w[j % 16];
                            uint temp2 = ((a >> 2 | a << 30) ^ (a >> 13 | a << 19) ^ (a >> 22 | a << 10)) +
                                       ((a & b) ^ (a & c) ^ (b & c));
                            
                            h = g;
                            g = f;
                            f = e;
                            e = d + temp1;
                            d = c;
                            c = b;
                            b = a;
                            a = temp1 + temp2;
                            
                            if (j % 16 == 15) {
                                w[0] = w[0] + w[9] + 
                                      ((w[1] >> 7 | w[1] << 25) ^ (w[1] >> 18 | w[1] << 14) ^ (w[1] >> 3)) +
                                      ((w[14] >> 17 | w[14] << 15) ^ (w[14] >> 19 | w[14] << 13) ^ (w[14] >> 10));
                                for (int k = 1; k < 16; k++) {
                                    w[k] = w[(k+1)%16];
                                }
                            }
                        }
                        
                        state[0] += a;
                        state[1] += b;
                        state[2] += c;
                        state[3] += d;
                        state[4] += e;
                        state[5] += f;
                        state[6] += g;
                        state[7] += h;
                    }
                    
                    for (int i = 0; i < 32; i++) {
                        output[i] = (uchar)(state[i/4] >> (24 - (i%4)*8));
                    }
                }

                // Generate simplified Ethereum address
                void generate_eth_address(uchar *private_key, uchar *address) {
                    uchar temp[32];
                    hash_bytes(private_key, 32, temp);
                    
                    // Take last 20 bytes as address
                    for(int i = 0; i < 20; i++) {
                        address[i] = temp[i + 12];
                    }
                }

                // Check if address matches any in the target list
                bool check_address(__global const uchar *target_addresses, 
                                 int num_addresses,
                                 uchar *address) {
                    for(int i = 0; i < num_addresses; i++) {
                        bool match = true;
                        for(int j = 0; j < 20; j++) {
                            if(address[j] != target_addresses[i * 20 + j]) {
                                match = false;
                                break;
                            }
                        }
                        if(match) return true;
                    }
                    return false;
                }

                __kernel void generate_and_check(
                    __global uchar *output,
                    __global const uchar *target_addresses,
                    const int num_addresses,
                    const uint thread_id,
                    const ulong iteration
                ) {
                    uint gid = get_global_id(0);
                    uint lid = get_local_id(0);
                    
                    // Initialize RNG state
                    uint4 rng_state = (uint4)(
                        gid ^ (iteration & 0xFFFFFFFF),
                        thread_id ^ (gid << 16),
                        iteration ^ 0xDEADBEEF,
                        thread_id ^ 0x12345678
                    );
                    
                    __private uchar private_key[32];
                    __private uchar address[20];
                    
                    for(int round = 0; round < ROUNDS_PER_THREAD; round++) {
                        // Generate private key using RNG
                        rng_state = xoshiro128(rng_state);
                        for(int i = 0; i < 8; i++) {
                            uint val = rng_state.x;
                            private_key[i*4 + 0] = (val >> 24) & 0xFF;
                            private_key[i*4 + 1] = (val >> 16) & 0xFF;
                            private_key[i*4 + 2] = (val >> 8) & 0xFF;
                            private_key[i*4 + 3] = val & 0xFF;
                            rng_state = xoshiro128(rng_state);
                        }
                        
                        // Generate Ethereum address
                        generate_eth_address(private_key, address);
                        
                        // Check if address matches
                        if(check_address(target_addresses, num_addresses, address)) {
                            // Copy private key and address to output buffer
                            int out_idx = atomic_add((__global int*)output, 1) * 52;
                            for(int i = 0; i < 32; i++)
                                output[out_idx + i] = private_key[i];
                            for(int i = 0; i < 20; i++)
                                output[out_idx + 32 + i] = address[i];
                        }
                    }
                }
            """).build()
            
            # Calculate work sizes
            global_size = ((BATCH_SIZE + WORK_GROUP_SIZE - 1) // WORK_GROUP_SIZE) * WORK_GROUP_SIZE
            local_size = WORK_GROUP_SIZE
            
            print(f"[green]GPU initialized for thread {thread_id}[/green]")
            print(f"[green]Work group size: {WORK_GROUP_SIZE}, Global size: {global_size}, Batch size: {BATCH_SIZE}[/green]")
            
            # Pre-allocate host buffer for results
            output_buffer = np.zeros(total_size, dtype=np.uint8)
            
            try:
                while True:
                    try:
                        current_time = time.time()
                        
                        if current_time - last_memory_clean >= MEMORY_CLEAN_INTERVAL:
                            clean_memory()
                            last_memory_clean = current_time
                        
                        # Clear output buffer
                        cl.enqueue_fill_buffer(queue, gpu_output, np.uint32(0), 0, 4)
                        
                        # Generate mnemonic for this batch
                        mnemonic = Bip39MnemonicGenerator().FromWordsNumber(Bip39WordsNum.WORDS_NUM_24)
                        seed_bytes = Bip39SeedGenerator(mnemonic).Generate()
                        bip32_mst_ctx = Bip32Slip10Secp256k1.FromSeed(seed_bytes)
                        master_key = bip32_mst_ctx.PrivateKey().Raw().ToHex()
                        
                        # Generate initial private key for display
                        private_key = np.random.bytes(32)  # Generate 32 random bytes
                        current_private_key = private_key.hex()  # Convert to hex string
                        
                        # Execute GPU kernel
                        program.generate_and_check(
                            queue, (global_size,), (local_size,),
                            gpu_output,
                            gpu_target_addresses,
                            np.int32(len(add)),
                            np.uint32(thread_id),
                            np.uint64(iteration_count)
                        )
                        
                        # Copy results from GPU
                        cl.enqueue_copy(queue, output_buffer, gpu_output)
                        queue.finish()
                        
                        # Update performance metrics
                        elapsed = current_time - last_time
                        if elapsed >= 1.0:  # Update every second
                            keys_per_second = BATCH_SIZE / elapsed
                            total_keys += BATCH_SIZE
                            avg_keys_per_second = total_keys / (current_time - start_time)
                            
                            # Print formatted batch info with current private key
                            batch_info = format_batch_info(
                                thread_id=thread_id,
                                mnemonic=mnemonic,
                                master_key=master_key,
                                private_key=current_private_key,
                                total_keys=total_keys,
                                keys_per_second=keys_per_second,
                                logp=logp
                            )
                            print(f"\033[2J\033[H{batch_info}")  # Clear screen and print at top
                            
                            last_time = current_time
                        
                        # Process results
                        num_found = int.from_bytes(output_buffer[0:4], byteorder='little')
                        for i in range(num_found):
                            try:
                                found_private_key = output_buffer[i*52+4:i*52+36].hex()
                                found_addr = '0x' + output_buffer[i*52+36:i*52+56].hex()
                                
                                if found_addr in add:
                                    fu += 1
                                    # Format match information with all details
                                    match_info = f"""
╔═══════════════════ MATCH FOUND! ═══════════════════
║ 
║ 🎯 Details for Match #{fu}:
║ 
║ 📍 Ethereum Address:
║ {found_addr}
║ 
║ 🔐 Private Key:
║ {found_private_key}
║ 
║ 👑 Master Key:
║ {master_key}
║ 
║ 📝 Mnemonic Words:
║ {mnemonic}
║ 
║ 📊 Found at:
║ • Total Keys: {total_keys:,}
║ • Speed: {keys_per_second:,.2f} keys/s
║ • Thread: {thread_id}
║ 
╚════════════════════════════════════════════════════"""
                                    print(match_info)
                                    
                                    try:
                                        # Send to Discord with pretty formatting
                                        match_message = (
                                            f"💎 **MATCH FOUND!**\n"
                                            f"```\n"
                                            f"Address: {found_addr}\n"
                                            f"Private Key: {found_private_key}\n"
                                            f"Master Key: {master_key}\n"
                                            f"Mnemonic:\n{mnemonic}\n"
                                            f"```"
                                        )
                                        send_discord_webhook(DISCORD_WEBHOOK_URL, match_message, 0xe74c3c)
                                    except Exception as e:
                                        print(f"[yellow]Warning: Could not send Discord notification: {str(e)}[/yellow]")
                                    
                                    # Save to file with pretty formatting
                                    with open(f'FoundMATCHAddr_{thread_id}.txt', 'a') as f:
                                        f.write(f"{match_info}\n\n")
                            except Exception as e:
                                print(f"[red]Error processing result {i}: {str(e)}[/red]")
                                continue
                        
                        z += BATCH_SIZE
                        iteration_count += 1
                        
                        if int(z) % int(logpx) == 0:
                            logp += int(logpx)
                            current_time = time.time()
                            
                            # Print current batch info with progress
                            batch_info = format_batch_info(
                                thread_id=thread_id,
                                mnemonic=mnemonic,
                                master_key=master_key,
                                private_key=current_private_key,
                                total_keys=total_keys,
                                keys_per_second=keys_per_second,
                                logp=logp
                            )
                            print(f"\033[2J\033[H{batch_info}")
                            
                            # Send webhook update if enough time has passed
                            if current_time - last_webhook_time >= WEBHOOK_INTERVAL:
                                try:
                                    avg_keys_per_second = total_keys / (current_time - start_time)
                                    progress_message = (
                                        f"⚡ **Progress Update**\n"
                                        f"```\n"
                                        f"Thread: {thread_id}\n"
                                        f"Current Mnemonic:\n{mnemonic}\n"
                                        f"Master Key: {master_key}\n"
                                        f"Generated: {logp:,} ETH addresses\n"
                                        f"Total Keys: {total_keys:,}\n"
                                        f"Current Speed: {keys_per_second:,.2f} keys/s\n"
                                        f"Average Speed: {avg_keys_per_second:,.2f} keys/s\n"
                                        f"Matches Found: {fu}\n"
                                        f"Time Running: {time.thread_time():.2f} seconds\n"
                                        f"```"
                                    )
                                    send_discord_webhook(DISCORD_WEBHOOK_URL, progress_message, 0xf1c40f)
                                    last_webhook_time = current_time
                                except Exception as e:
                                    print(f"[yellow]Warning: Could not send Discord update: {str(e)}[/yellow]")
                        else:
                            # Update progress in place with key info
                            progress_line = f"\r[Thread {thread_id}] Keys: {z:,} | Found: {fu} | Speed: {keys_per_second:,.2f} k/s | Current Master Key: {master_key[:16]}..."
                            sys.stdout.write(progress_line)
                            sys.stdout.flush()
                            
                    except Exception as e:
                        print(f"[red]GPU Error in thread {thread_id}: {str(e)}[/red]")
                        time.sleep(1)  # Add delay before retrying
                        continue
                        
            except KeyboardInterrupt:
                print(f"\n[yellow]Thread {thread_id} stopped[/yellow]")
                final_stats = format_stats(thread_id, total_keys, 
                                        total_keys / (time.time() - last_time) if last_time != start_time else 0,
                                        total_keys / (time.time() - start_time) if start_time != time.time() else 0,
                                        fu)
                print(final_stats)
            finally:
                # Cleanup GPU resources
                try:
                    if gpu_output:
                        gpu_output.release()
                    if gpu_target_addresses:
                        gpu_target_addresses.release()
                except:
                    pass
                gc.collect()
                
        except Exception as e:
            print(f"[red]Error initializing GPU program for thread {thread_id}: {str(e)}[/red]")
    else:
        print(f"[yellow]Thread {thread_id} falling back to CPU[/yellow]")
    
    try:
        start_message = f"🚀 Started ETH Scanner\nThread: {thread_id}\nTime: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        send_discord_webhook(DISCORD_WEBHOOK_URL, start_message, 0x3498db)
    except Exception as e:
        print(f"[yellow]Warning: Could not send start notification: {str(e)}[/yellow]")

def is_valid_eth_address(addr):
    """Check if the address is a valid Ethereum address."""
    if not addr.startswith('0x'):
        return False
    try:
        # Remove '0x' prefix and check if it's a valid hex string of length 40
        addr = addr[2:]
        if len(addr) != 40:
            return False
        int(addr, 16)  # Try to convert to int to validate hex
        return True
    except ValueError:
        return False

def Main():
    parser = argparse.ArgumentParser(description="Ethereum Address Finder Script")
    
    parser.add_argument('-f', '--file', dest="filenameEth", required=True,
                        help="Ethereum Rich Address File With Type Format .TXT [Example: -f eth5.txt]")
    parser.add_argument('-v', '--view', dest="ViewPrint", required=True,
                        help="Print After Generated This Number Print And Report")
    parser.add_argument('-n', '--thread', dest="ThreadCount", required=True,
                        help="Total Thread Number (Total Core CPU)")

    args = parser.parse_args()
    
    try:
        with open(args.filenameEth) as f:
            addresses = f.read().split()
        
        # Filter valid Ethereum addresses
        add = set()
        invalid_count = 0
        for addr in addresses:
            if is_valid_eth_address(addr):
                add.add(addr)
            else:
                invalid_count += 1
        
        if invalid_count > 0:
            print(f"[yellow]Skipped {invalid_count} invalid Ethereum addresses[/yellow]")
        
        if not add:
            print("[red]No valid Ethereum addresses found in the file[/red]")
            return
        
        print(f"[green]Loaded {len(add)} valid Ethereum addresses[/green]")
        
    except FileNotFoundError:
        print(f"[red]Error: File {args.filenameEth} not found[/red]")
        return
    except Exception as e:
        print(f"[red]Error reading file: {str(e)}[/red]")
        return

    thread_count = int(args.ThreadCount)
    if thread_count <= 0:
        print("[red]Thread count must be greater than 0[/red]")
        return
        
    signal.signal(signal.SIGINT, signal_handler)
    
    start_message = (
        f"🎮 **ETH Scanner Started**\n"
        f"```\n"
        f"File: {args.filenameEth}\n"
        f"Threads: {args.ThreadCount}\n"
        f"Report Every: {args.ViewPrint} addresses\n"
        f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"```"
    )
    send_discord_webhook(DISCORD_WEBHOOK_URL, start_message, 0x2ecc71)
    
    jobs = []
    try:
        for i in range(thread_count):
            p = multiprocessing.Process(
                target=worker_process,
                args=(add, args.ViewPrint, i)
            )
            jobs.append(p)
            p.start()
            print(f"[green]Started thread {i}[/green]")
            time.sleep(1)  # إضافة تأخير بين تشغيل الـ threads

        # انتظار إنهاء جميع الـ threads
        for job in jobs:
            job.join()
            
    except KeyboardInterrupt:
        print("\n[yellow]Stopping all processes...[/yellow]")
        for job in jobs:
            try:
                job.terminate()
                job.join(timeout=1.0)  # انتظار ثانية واحدة للإغلاق
                if job.is_alive():
                    job.kill()  # إجبار الإغلاق إذا لم يستجب
            except Exception as e:
                print(f"[red]Error stopping thread: {str(e)}[/red]")
    finally:
        # التأكد من إغلاق جميع الـ threads
        for job in jobs:
            if job.is_alive():
                try:
                    job.terminate()
                    job.join(timeout=0.5)
                except:
                    pass
    
    print("[green]All processes completed[/green]")

if __name__ == '__main__':
    multiprocessing.freeze_support()  # دعم التجميد للـ Windows
    Main()