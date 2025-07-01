let SessionLoad = 1
let s:so_save = &g:so | let s:siso_save = &g:siso | setg so=0 siso=0 | setl so=-1 siso=-1
let v:this_session=expand("<sfile>:p")
silent only
silent tabonly
cd ~/Documentos/ACAD/MAESTRÍA/Cuatrimestre_MarzoAgosto_2025/CómputoEvolutivo/ProgramaciónGenética.TEX
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
let s:shortmess_save = &shortmess
if &shortmess =~ 'A'
  set shortmess=aoOA
else
  set shortmess=aoO
endif
badd +8 ProgramaciónEvolutiva.tex
badd +64 bibliografia.bib
badd +176 01.-EvolutionaryComputing.tex
badd +137 Macros.tex
badd +4 Ejemplo/DEAP-GP.qmd
argglobal
%argdel
tabnew +setlocal\ bufhidden=wipe
tabnew +setlocal\ bufhidden=wipe
tabnew +setlocal\ bufhidden=wipe
tabnew +setlocal\ bufhidden=wipe
tabrewind
edit 01.-EvolutionaryComputing.tex
let s:save_splitbelow = &splitbelow
let s:save_splitright = &splitright
set splitbelow splitright
wincmd _ | wincmd |
split
1wincmd k
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
exe '1resize ' . ((&lines * 155 + 84) / 169)
exe '2resize ' . ((&lines * 10 + 84) / 169)
argglobal
balt Macros.tex
setlocal foldmethod=expr
setlocal foldexpr=nvim_treesitter#foldexpr()
setlocal foldmarker={{{,}}}
setlocal foldignore=#
setlocal foldlevel=0
setlocal foldminlines=1
setlocal foldnestmax=20
setlocal foldenable
2
sil! normal! zo
22
sil! normal! zo
23
sil! normal! zo
29
sil! normal! zo
30
sil! normal! zo
31
sil! normal! zo
101
sil! normal! zo
109
sil! normal! zo
117
sil! normal! zo
119
sil! normal! zo
124
sil! normal! zo
125
sil! normal! zo
126
sil! normal! zo
173
sil! normal! zo
174
sil! normal! zo
179
sil! normal! zo
let s:l = 175 - ((174 * winheight(0) + 77) / 155)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 175
normal! 0
wincmd w
argglobal
enew
balt 01.-EvolutionaryComputing.tex
setlocal foldmethod=manual
setlocal foldexpr=nvim_treesitter#foldexpr()
setlocal foldmarker={{{,}}}
setlocal foldignore=#
setlocal foldlevel=0
setlocal foldminlines=1
setlocal foldnestmax=20
setlocal foldenable
wincmd w
exe '1resize ' . ((&lines * 155 + 84) / 169)
exe '2resize ' . ((&lines * 10 + 84) / 169)
tabnext
edit Ejemplo/DEAP-GP.qmd
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
let s:l = 5 - ((4 * winheight(0) + 83) / 166)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 5
normal! 0
tabnext
edit ProgramaciónEvolutiva.tex
argglobal
balt 01.-EvolutionaryComputing.tex
setlocal foldmethod=expr
setlocal foldexpr=nvim_treesitter#foldexpr()
setlocal foldmarker={{{,}}}
setlocal foldignore=#
setlocal foldlevel=0
setlocal foldminlines=1
setlocal foldnestmax=20
setlocal foldenable
let s:l = 8 - ((7 * winheight(0) + 83) / 166)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 8
normal! 013|
tabnext
edit bibliografia.bib
argglobal
balt 01.-EvolutionaryComputing.tex
setlocal foldmethod=expr
setlocal foldexpr=nvim_treesitter#foldexpr()
setlocal foldmarker={{{,}}}
setlocal foldignore=#
setlocal foldlevel=0
setlocal foldminlines=1
setlocal foldnestmax=20
setlocal foldenable
60
sil! normal! zo
let s:l = 64 - ((63 * winheight(0) + 83) / 166)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 64
normal! 025|
tabnext
edit Macros.tex
argglobal
balt 01.-EvolutionaryComputing.tex
setlocal foldmethod=expr
setlocal foldexpr=nvim_treesitter#foldexpr()
setlocal foldmarker={{{,}}}
setlocal foldignore=#
setlocal foldlevel=0
setlocal foldminlines=1
setlocal foldnestmax=20
setlocal foldenable
let s:l = 38 - ((37 * winheight(0) + 83) / 166)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 38
normal! 071|
tabnext 1
if exists('s:wipebuf') && len(win_findbuf(s:wipebuf)) == 0 && getbufvar(s:wipebuf, '&buftype') isnot# 'terminal'
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20
let &shortmess = s:shortmess_save
let s:sx = expand("<sfile>:p:r")."x.vim"
if filereadable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &g:so = s:so_save | let &g:siso = s:siso_save
set hlsearch
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
