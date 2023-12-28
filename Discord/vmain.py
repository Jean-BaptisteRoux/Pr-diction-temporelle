import discord
from discord.ext import commands

bot = commands.Bot(command_prefix='!',intents=discord.Intents.all())

@bot.command()
async def fichier(ctx):
    with open('/home/Astra_world/Astra_investissement/Version_1.1/logs2.txt', 'r') as f:
        contenu = f.readlines()[-24:]
    for i in contenu:
        await ctx.send(i)
    await ctx.send("FIN")
        
@bot.command()
async def mouv(ctx):
    with open('/home/Astra_world/Astra_investissement/Version_1.1/logs3.txt', 'r') as f:
        try:
            contenu = f.readlines()[-24:]
        except IndexError:
            contenu = f.readlines()
    for i in contenu:
        await ctx.send(i)
    await ctx.send("FIN")
@bot.command()
async def ordre(ctx):
    with open('/home/Astra_world/Astra_investissement/Version_1.1/order.txt', 'r') as f:
        try:
            contenu = f.readlines()[-24:]
        except IndexError:
            contenu = f.readlines()
    for i in contenu:
        await ctx.send(i)
    await ctx.send("FIN")


bot.run('MTA1ODc3ODI2MzczODEzMDQ3Mw.GaK142.qoLxXLQ0c_ssSsxr0lVrIza72zi0TUdNPpdFqE')
