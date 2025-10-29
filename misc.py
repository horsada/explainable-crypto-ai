from excrypto.utils.hash_debug import explain_diff


if __name__ == '__main__':
    explain_diff("configs/features/default.yaml", "configs/labels/fh_24_cls.yaml",
             snapshot="2019-01-01_to_2019-12-31", syms="BTC/USDT", timeframe="1h")