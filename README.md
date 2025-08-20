![](./src/frinZlogo.png)
<img src="./src/frinZlogo.png"> <img src="./src/frinZlogo.png">

# frinZrs
Rust version of frinZ.py  
https://github.com/M-AKIMOTOO/frinZ.py

# Compile
```
cargo run --bin frinZ --release  
cargo run --bin frinZ-rs --release
```
のどちらか．どちらも同じプログラム．

# Install  
```
cargo install --bin frinZ --path .  
cargo install --bin frinZ-rs --path .
```
--path の直後にピリオドがあることに注意．~/.cargo/bin にインストールされるはず．コンパイルすると windows で利用しているセキュリティソフトに引っかかった．

# Note
~~frinZrs は frinZ.py を Rust で書き換えたプログラムだが，微妙に相関出力が違う（0.1 % くらい）．その原因は Python と Rust でバイナリの精度が微妙に違うようで，Python は下 6 桁までデコードするが，Rust は下 7--8 桁までデコードする．float32 や float64 でデコードして比較をしたが結果は変わらずだった．もちろん FFT のライブラリーが違うことも考えられる．frinZ.py は scipy.fft で，Rust は rustfft を用いている．~~
大嘘でござる。FFT で DC 成分をカット、つまり 0+0j に置換していなかっただけ。それでも積分時間が長いと数値誤差（C++ と Python は下 6 桁までバイナリをデコードし、Rust は下 7 桁までデコードすることがある）が顕著となる。それでも frinZrs は frinZ.py や frinZsearch と 0.1 % くらいしか違わない。
