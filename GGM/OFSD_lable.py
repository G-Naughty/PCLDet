import os
lable_path="/home/disk/OFSD/trainval/labelTxt/"
provide_name=['Nimitz Aircraft Carrier', 'Barracks Ship', 'Container Ship', 'Fishing Vessel', 'Henry J. Kaiser-class replenishment oiler',
               'Other Warship', 'Yacht', 'Freedom-class littoral combat ship', 'Arleigh Burke-class Destroyer', 'Lewis and Clark-class dry cargo ship',
               'Towing vessel', 'unknown', 'Powhatan-class tugboat', 'Barge', '055-destroyer', '052D-destroyer', 'USNS Bob Hope', 'USNS Montford Point',
               'Bunker', 'Ticonderoga-class cruiser', 'Oliver Hazard Perry-class frigate', 'Sacramento-class fast combat support ship', 'Submarine',
               'Emory S. Land-class submarine tender', 'Hatakaze-class destroyer', 'Murasame-class destroyer', 'Whidbey Island-class dock landing ship',
               'Hiuchi-class auxiliary multi-purpose support ship', 'USNS Spearhead', 'Hyuga-class helicopter destroyer', 'Akizuki-class destroyer',
               'Bulk carrier', 'Kongō-class destroyer', 'Northampton-class tug', 'Sand Carrier', 'Iowa-class battle ship', 'Independence-class littoral combat ship',
               'Tarawa-class amphibious assault ship', 'Cyclone-class patrol ship', 'Wasp-class amphibious assault ship', '074-landing ship',
               '056-corvette', '721-transport boat', '037Ⅱ-missile boat', 'Traffic boat', '037-submarine chaser', 'unknown auxiliary ship',
               '072Ⅲ-landing ship', '636-hydrographic survey ship', '272-icebreaker', '529-Minesweeper', '053H2G-frigate', '909A-experimental ship',
               '909-experimental ship', '037-hospital ship', 'Tuzhong Class Salvage Tug', '022-missile boat', '051-destroyer', '054A-frigate',
               '082Ⅱ-Minesweeper', '053H1G-frigate', 'Tank ship', 'Hatsuyuki-class destroyer', 'Sugashima-class minesweepers', 'YG-203 class yard gasoline oiler',
               'Hayabusa-class guided-missile patrol boats', 'JS Chihaya', 'Kurobe-class training support ship', 'Abukuma-class destroyer escort',
               'Uwajima-class minesweepers', 'Osumi-class landing ship', 'Hibiki-class ocean surveillance ships', 'JMSDF LCU-2001 class utility landing crafts',
               'Asagiri-class Destroyer',  'Uraga-class Minesweeper Tender', 'Tenryu-class training support ship', 'YW-17 Class Yard Water',
               'Izumo-class helicopter destroyer', 'Towada-class replenishment oilers', 'Takanami-class destroyer', 'YO-25 class yard oiler',
               '891A-training ship', '053H3-frigate', '922A-Salvage lifeboat', '680-training ship', '679-training ship',
               '072A-landing ship', '072Ⅱ-landing ship', 'Mashu-class replenishment oilers', '903A-replenishment ship', '815A-spy ship',
               '901-fast combat support ship', 'Xu Xiake barracks ship', 'San Antonio-class amphibious transport dock', '908-replenishment ship',
               '052B-destroyer', '904-general stores issue ship', '051B-destroyer', '925-Ocean salvage lifeboat', '904B-general stores issue ship',
               '625C-Oceanographic Survey Ship', '071-amphibious transport dock', '052C-destroyer', '635-hydrographic Survey Ship',
               '926-submarine support ship', '917-lifeboat', 'Mercy-class hospital ship', 'Lewis B. Puller-class expeditionary mobile base ship',
               'Avenger-class mine countermeasures ship', 'Zumwalt-class destroyer', '920-hospital ship',
               '052-destroyer', '054-frigate', '051C-destroyer', '903-replenishment ship', '073-landing ship', '074A-landing ship',
               'North Transfer 990', '001-aircraft carrier', '905-replenishment ship', 'Hatsushima-class minesweeper',
               'Forrestal-class Aircraft Carrier', 'Kitty Hawk class aircraft carrier', 'Blue Ridge class command ship', '081-Minesweeper',
               '648-submarine repair ship', '639A-Hydroacoustic measuring ship', 'JS Kurihama', 'JS Suma',
               'Futami-class hydro-graphic survey ships', 'Yaeyama-class minesweeper',
               '815-spy ship', 'Sovremenny-class destroyer']
true_name=[]
dif_between_provide_true=[]
new_name=[]
file_names = os.listdir(lable_path)
for file_name in file_names:
    with open(lable_path+file_name, 'r') as f:
        for line in f:
            items = line.split(',')
            if len(items) >= 9:
                lable=items[8]
                if lable not in true_name:
                    true_name.append(lable)
                    new_lable_name = lable.strip().split(' ')
                    new_lable_name = '_'.join(new_lable_name)
                    new_name.append(new_lable_name)
                    if lable not in provide_name:
                        dif_between_provide_true.append(lable)
print("true name:\n", true_name, "\n")
print("new name:", len(new_name), "\n", new_name, "\n")
print("difference between provide name and true name:\n", dif_between_provide_true)