# frinZrs
Rust version of frinZ.py

# Install 
```
cargo run --bin frinZ --release  
cargo run --bin frinZ-rs --release
```
のどちらか．どちらも同じプログラム．

# Note
frinZrs は frinZ.py を Rust で書き換えたプログラムだが，微妙に（0.1 % くらい）相関出力が違う．その原因は Python と Rust でバイナリの精度が微妙に違うようで，Python は下 6 桁までデコードするが，Rust は下 7--8 桁までデコードする．float32 や float64 でデコードして比較をしたが結果は変わらずだった．


