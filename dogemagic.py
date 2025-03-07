import ctypes
import time
import argparse
import multiprocessing
from bip_utils import Bip32Slip10Secp256k1, Bip39MnemonicGenerator, Bip39SeedGenerator, Bip39WordsNum, P2PKHAddr
from rich import print


def Main():
    # Setup argparse for command-line options
    parser = argparse.ArgumentParser(description="Dogecoin Address Finder Script")

    parser.add_argument('-f', '--file', dest="filenameDoge", required=True,
                        help="Dogecoin Rich Address File With Type Format .TXT [Example: -f doge5.txt]")
    parser.add_argument('-v', '--view', dest="ViewPrint", required=True,
                        help="Print After Generated This Number Print And Report")
    parser.add_argument('-n', '--thread', dest="ThreadCount", required=True,
                        help="Total Thread Number (Total Core CPU)")

    # Parse arguments
    args = parser.parse_args()
    filename = args.filenameDoge
    logpx = args.ViewPrint
    thco = args.ThreadCount

    # ----------------------- START ------------------------------ #
    with open(filename) as f:
        add = f.read().split()
    add = set(add)

    z = 0
    fu = 0
    logp = 0

    while True:
        z += 1
        ctypes.windll.kernel32.SetConsoleTitleW(f"MATCH:{fu} SCAN:{z}")
        mnemonic = Bip39MnemonicGenerator().FromWordsNumber(Bip39WordsNum.WORDS_NUM_24)
        seed_bytes = Bip39SeedGenerator(mnemonic).Generate()
        bip32_mst_ctx = Bip32Slip10Secp256k1.FromSeed(seed_bytes)
        MasterKey = bip32_mst_ctx.PrivateKey().Raw().ToHex()
        bip32_der_ctx = bip32_mst_ctx.DerivePath("m/44'/3'/0'/0/0")
        PrivateKeyBytes = bip32_der_ctx.PrivateKey().Raw().ToHex()

        # Encode the public key into a Dogecoin address with the mainnet version byte
        addr = P2PKHAddr.EncodeKey(
            bip32_der_ctx.PublicKey().KeyObject(),
            net_ver=bytes([0x1E])  # Convert the network version to bytes
        )
        Words24 = str(mnemonic)

        if addr in add:
            fu += 1
            print(f"[green1][+] MATCH ADDRESS FOUND IN LIST IMPORTED :[/green1] [white]{addr}[/white]")
            print(
                f"PrivateKey (Byte) : [green1]{PrivateKeyBytes}[/green1]\n[gold1]{mnemonic}[/gold1]\n[red1]MasterKey (Byte) : [/red1][green1]{MasterKey}[/green1]")
            with open('FoundMATCHAddr_DOGE.txt', 'a') as f:
                f.write(
                    f"{addr}\n{PrivateKeyBytes}\n{mnemonic}\n{MasterKey}\n------------------------- MMDRZA.Com -------------------\n")
                f.close()
        elif int(z) % int(logpx) == 0:
            logp += int(logpx)
            print(
                f"[red][[green1]+[/green1]][GENERATED[white] {logp}[/white] DOGE ADDR][WITH [white]{thco} THREAD[/white]][sK/Th:[white]{time.thread_time()}[/white]][/red]\n[red][[green1]{PrivateKeyBytes.upper()}[/green1]][/red]")
            print(
                f"[red][MasterKey : [white]{MasterKey.upper()}[/white]][/red]\n[white on red3][MNEMONIC : {Words24[0:64]}...][/white on red3]")
        else:
            print(
                f"[red][-][ GENERATED [cyan]{z}[/cyan] DOGE ADDR ][FOUND:[white]{fu}[/white]][THREAD:[cyan]{thco}[/cyan]][/red]",
                end="\r")


jobs = []
if __name__ == '__main__':
    m = multiprocessing.Process(target=Main)
    jobs.append(m)
    m.start()
