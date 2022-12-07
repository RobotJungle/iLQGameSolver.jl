using LinearAlgebra
"""
Get Player i's start and final index
"""
function getPlayerIdx(game, i)
    nx = game.nx
    nu = game.nu
    Nplayer = game.Nplayer
    Nx = Nplayer*nx   

    Nxi = 1+(i-1)*Nx     # Player i's state start index
    Nxf = i*Nx           # Player i's state final index
    nui = 1+(i-1)*nu     # Player i's control start index
    nuf = i*nu           # Player i's control final index
    return Nxi,Nxf,nui,nuf
end