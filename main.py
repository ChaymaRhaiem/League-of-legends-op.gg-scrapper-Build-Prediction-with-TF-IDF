import win_items




if __name__ == '__main__':

    """
    #win_items.get(lane="mid",name="ahri")
    print(summoner)
    for i in range(len(summoner)):
        champions, killdeathassist, results, mode = win_items.get_items(summoner)
        build =win_items.get_item_name(summoner)
        dataframes = win_items.list_to_df(summoner,build,champions,killdeathassist,results,mode)
        print(dataframes)
"""
    summoner = win_items.get_summoner_name()
    summoner_profile = win_items.summoner_url(summoner)
    for i in range(len(summoner_profile)):
        print(summoner_profile[i])
        win_items.main(summoner_profile)



