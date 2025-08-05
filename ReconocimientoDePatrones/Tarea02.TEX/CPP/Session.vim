let SessionLoad = 1
let s:so_save = &g:so | let s:siso_save = &g:siso | setg so=0 siso=0 | setl so=-1 siso=-1
let v:this_session=expand("<sfile>:p")
silent only
silent tabonly
cd ~/Documentos/ACAD/MAESTRÍA/Cuatrimestre_MarzoAgosto_2025/ReconocimientoDePatrones/Tarea02.TEX/CPP
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
let s:shortmess_save = &shortmess
if &shortmess =~ 'A'
  set shortmess=aoOA
else
  set shortmess=aoO
endif
badd +45 src/MatrizCaracteristicas.cxx
badd +6 src/MatrizCaracteristicas_main.cpp
badd +7 src/MatrizCaracteristicas.hxx
argglobal
%argdel
edit src/MatrizCaracteristicas.cxx
let s:save_splitbelow = &splitbelow
let s:save_splitright = &splitright
set splitbelow splitright
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd w
let &splitbelow = s:save_splitbelow
let &splitright = s:save_splitright
wincmd t
let s:save_winminheight = &winminheight
let s:save_winminwidth = &winminwidth
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe 'vert 1resize ' . ((&columns * 30 + 133) / 266)
exe 'vert 2resize ' . ((&columns * 235 + 133) / 266)
argglobal
enew
file NvimTree_1
balt src/MatrizCaracteristicas.hxx
setlocal foldmethod=syntax
setlocal foldexpr=nvim_treesitter#foldexpr()
setlocal foldmarker={{{,}}}
setlocal foldignore=#
setlocal foldlevel=0
setlocal foldminlines=1
setlocal foldnestmax=20
setlocal nofoldenable
wincmd w
argglobal
setlocal foldmethod=expr
setlocal foldexpr=nvim_treesitter#foldexpr()
setlocal foldmarker={{{,}}}
setlocal foldignore=#
setlocal foldlevel=0
setlocal foldminlines=1
setlocal foldnestmax=20
setlocal foldenable
3
sil! normal! zo
4
sil! normal! zo
8
sil! normal! zo
14
sil! normal! zo
16
sil! normal! zo
26
sil! normal! zo
26
sil! normal! zo
30
sil! normal! zo
34
sil! normal! zo
47
sil! normal! zo
50
sil! normal! zo
59
sil! normal! zo
61
sil! normal! zo
let s:l = 9 - ((8 * winheight(0) + 33) / 66)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 9
normal! 06|
wincmd w
2wincmd w
exe 'vert 1resize ' . ((&columns * 30 + 133) / 266)
exe 'vert 2resize ' . ((&columns * 235 + 133) / 266)
tabnext 1
if exists('s:wipebuf') && len(win_findbuf(s:wipebuf)) == 0 && getbufvar(s:wipebuf, '&buftype') isnot# 'terminal'
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20
let &shortmess = s:shortmess_save
let &winminheight = s:save_winminheight
let &winminwidth = s:save_winminwidth
let s:sx = expand("<sfile>:p:r")."x.vim"
if filereadable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &g:so = s:so_save | let &g:siso = s:siso_save
set hlsearch
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
