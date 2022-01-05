import summoner as summoner

import win_items
import data



if __name__ == '__main__':
    op = str(input("sup : ").capitalize())
    if op =="scraping":

        summoner = win_items.get_summoner_name()
        summoner_profile = win_items.summoner_url(summoner)
        for i in range(len(summoner_profile)):
            print(summoner_profile[i])
            win_items.main(summoner_profile)
    else:
        recc = str(input("Item recommendation?: ").capitalize())
        data.get_recommendations(item=recc, cosine_sim=data.cosine_sim)

