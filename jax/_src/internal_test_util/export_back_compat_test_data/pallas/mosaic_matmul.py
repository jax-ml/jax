# Copyright 2024 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
from numpy import array, float32


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_09_24 = dict(
    testdata_version=1,
    platform='tpu',
    custom_call_targets=['tpu_custom_call'],
    serialized_date=datetime.date(2024, 9, 24),
    inputs=(),
    expected_outputs=(array([[   90458.2  ,    90470.875,    90480.85 ,    90491.1  ,
           90500.945,    90510.945,    90521.19 ,    90530.95 ,
           90540.78 ,    90551.16 ,    90560.67 ,    90570.734,
           90580.73 ,    90590.586,    90600.66 ,    90610.61 ],
       [  643341.75 ,   643434.25 ,   643509.75 ,   643587.1  ,
          643660.1  ,   643735.94 ,   643813.5  ,   643886.   ,
          643960.6  ,   644039.5  ,   644110.25 ,   644186.75 ,
          644262.5  ,   644336.06 ,   644412.9  ,   644488.3  ],
       [ 1196323.2  ,  1196495.6  ,  1196636.8  ,  1196781.   ,
         1196917.5  ,  1197059.   ,  1197203.9  ,  1197339.2  ,
         1197478.5  ,  1197625.8  ,  1197757.8  ,  1197900.5  ,
         1198042.   ,  1198179.4  ,  1198323.1  ,  1198464.   ],
       [ 1749075.5  ,  1749327.8  ,  1749534.4  ,  1749746.   ,
         1749945.5  ,  1750152.8  ,  1750365.1  ,  1750563.   ,
         1750767.2  ,  1750983.1  ,  1751176.2  ,  1751385.5  ,
         1751592.8  ,  1751793.8  ,  1752004.2  ,  1752210.8  ],
       [ 2302500.5  ,  2302832.5  ,  2303104.8  ,  2303383.5  ,
         2303646.2  ,  2303919.5  ,  2304199.   ,  2304459.8  ,
         2304728.5  ,  2305013.   ,  2305267.2  ,  2305542.8  ,
         2305816.2  ,  2306081.   ,  2306358.5  ,  2306630.5  ],
       [ 2855440.2  ,  2855852.2  ,  2856190.2  ,  2856535.5  ,
         2856861.5  ,  2857200.5  ,  2857547.2  ,  2857870.8  ,
         2858204.5  ,  2858557.   ,  2858872.5  ,  2859214.5  ,
         2859553.2  ,  2859882.   ,  2860226.   ,  2860563.5  ],
       [ 3407472.   ,  3407964.2  ,  3408367.5  ,  3408780.2  ,
         3409169.5  ,  3409574.2  ,  3409988.5  ,  3410374.5  ,
         3410772.8  ,  3411194.   ,  3411570.5  ,  3411978.8  ,
         3412383.5  ,  3412776.   ,  3413186.8  ,  3413590.   ],
       [ 3959847.5  ,  3960419.   ,  3960888.   ,  3961367.8  ,
         3961820.2  ,  3962291.   ,  3962772.5  ,  3963221.2  ,
         3963684.8  ,  3964174.5  ,  3964612.   ,  3965086.8  ,
         3965557.2  ,  3966013.2  ,  3966491.   ,  3966959.5  ],
       [ 4515869.5  ,  4516521.5  ,  4517056.   ,  4517602.5  ,
         4518118.   ,  4518654.5  ,  4519203.   ,  4519715.   ,
         4520243.   ,  4520801.   ,  4521300.   ,  4521841.   ,
         4522377.5  ,  4522897.   ,  4523441.5  ,  4523975.5  ],
       [ 5061659.   ,  5062390.   ,  5062990.   ,  5063603.5  ,
         5064182.   ,  5064784.5  ,  5065401.   ,  5065975.   ,
         5066567.5  ,  5067194.   ,  5067754.   ,  5068362.   ,
         5068964.   ,  5069547.   ,  5070159.   ,  5070758.5  ],
       [ 5621329.   ,  5622141.   ,  5622806.5  ,  5623487.5  ,
         5624129.5  ,  5624797.   ,  5625481.   ,  5626118.   ,
         5626775.   ,  5627470.5  ,  5628092.   ,  5628765.   ,
         5629433.   ,  5630080.5  ,  5630758.5  ,  5631424.   ],
       [ 6172820.5  ,  6173712.   ,  6174443.   ,  6175191.   ,
         6175896.5  ,  6176630.   ,  6177381.   ,  6178080.5  ,
         6178803.   ,  6179566.   ,  6180248.5  ,  6180988.5  ,
         6181722.   ,  6182432.5  ,  6183177.5  ,  6183908.   ],
       [ 6723343.5  ,  6724315.   ,  6725111.5  ,  6725927.   ,
         6726696.   ,  6727495.5  ,  6728313.5  ,  6729076.5  ,
         6729864.   ,  6730696.   ,  6731440.   ,  6732246.   ,
         6733046.   ,  6733820.5  ,  6734632.   ,  6735428.5  ],
       [ 7280537.   ,  7281587.5  ,  7282449.5  ,  7283331.5  ,
         7284163.5  ,  7285029.   ,  7285914.   ,  7286739.   ,
         7287591.   ,  7288492.   ,  7289297.   ,  7290169.5  ,
         7291035.   ,  7291873.5  ,  7292752.   ,  7293614.   ],
       [ 7828292.   ,  7829423.   ,  7830350.   ,  7831299.5  ,
         7832194.5  ,  7833125.5  ,  7834078.5  ,  7834966.   ,
         7835883.   ,  7836852.   ,  7837718.   ,  7838657.   ,
         7839588.   ,  7840490.   ,  7841436.   ,  7842363.5  ],
       [ 8384808.5  ,  8386019.5  ,  8387012.5  ,  8388029.5  ,
         8388988.   ,  8389985.   ,  8391005.   ,  8391956.   ,
         8392937.   ,  8393974.   ,  8394902.   ,  8395907.   ,
         8396904.   ,  8397870.   ,  8398882.   ,  8399875.   ],
       [ 8928697.   ,  8929987.   ,  8931044.   ,  8932126.   ,
         8933146.   ,  8934208.   ,  8935294.   ,  8936306.   ,
         8937351.   ,  8938455.   ,  8939443.   ,  8940514.   ,
         8941574.   ,  8942604.   ,  8943682.   ,  8944738.   ],
       [ 9501496.   ,  9502866.   ,  9503990.   ,  9505141.   ,
         9506226.   ,  9507354.   ,  9508508.   ,  9509584.   ,
         9510695.   ,  9511870.   ,  9512919.   ,  9514058.   ,
         9515186.   ,  9516279.   ,  9517425.   ,  9518549.   ],
       [10055416.   , 10056868.   , 10058060.   , 10059279.   ,
        10060428.   , 10061624.   , 10062848.   , 10063988.   ,
        10065166.   , 10066410.   , 10067522.   , 10068729.   ,
        10069925.   , 10071083.   , 10072298.   , 10073489.   ],
       [10595886.   , 10597416.   , 10598672.   , 10599958.   ,
        10601170.   , 10602431.   , 10603721.   , 10604923.   ,
        10606164.   , 10607477.   , 10608650.   , 10609922.   ,
        10611182.   , 10612404.   , 10613684.   , 10614940.   ],
       [11135804.   , 11137412.   , 11138732.   , 11140083.   ,
        11141357.   , 11142682.   , 11144038.   , 11145302.   ,
        11146606.   , 11147985.   , 11149218.   , 11150554.   ,
        11151880.   , 11153164.   , 11154509.   , 11155829.   ],
       [11686791.   , 11688480.   , 11689864.   , 11691282.   ,
        11692618.   , 11694007.   , 11695430.   , 11696756.   ,
        11698124.   , 11699570.   , 11700864.   , 11702265.   ,
        11703656.   , 11705003.   , 11706414.   , 11707799.   ],
       [12263420.   , 12265190.   , 12266642.   , 12268128.   ,
        12269529.   , 12270986.   , 12272478.   , 12273868.   ,
        12275303.   , 12276820.   , 12278176.   , 12279646.   ,
        12281104.   , 12282516.   , 12283996.   , 12285446.   ],
       [12821178.   , 12823029.   , 12824548.   , 12826102.   ,
        12827567.   , 12829092.   , 12830652.   , 12832106.   ,
        12833606.   , 12835192.   , 12836610.   , 12838148.   ,
        12839673.   , 12841150.   , 12842699.   , 12844217.   ],
       [13362964.   , 13364895.   , 13366479.   , 13368100.   ,
        13369628.   , 13371218.   , 13372846.   , 13374362.   ,
        13375927.   , 13377582.   , 13379061.   , 13380665.   ,
        13382256.   , 13383796.   , 13385411.   , 13386995.   ],
       [13902882.   , 13904890.   , 13906538.   , 13908225.   ,
        13909815.   , 13911470.   , 13913163.   , 13914740.   ,
        13916368.   , 13918090.   , 13919629.   , 13921298.   ,
        13922952.   , 13924556.   , 13926236.   , 13927884.   ],
       [14443848.   , 14445934.   , 14447646.   , 14449398.   ,
        14451050.   , 14452769.   , 14454528.   , 14456166.   ,
        14457858.   , 14459647.   , 14461246.   , 14462979.   ,
        14464698.   , 14466363.   , 14468108.   , 14469820.   ],
       [15024406.   , 15026576.   , 15028355.   , 15030176.   ,
        15031893.   , 15033679.   , 15035507.   , 15037210.   ,
        15038968.   , 15040828.   , 15042490.   , 15044291.   ,
        15046077.   , 15047808.   , 15049621.   , 15051400.   ],
       [15586096.   , 15588347.   , 15590193.   , 15592082.   ,
        15593863.   , 15595716.   , 15597613.   , 15599380.   ,
        15601204.   , 15603133.   , 15604856.   , 15606726.   ,
        15608579.   , 15610375.   , 15612257.   , 15614103.   ],
       [16130043.   , 16132373.   , 16134285.   , 16136242.   ,
        16138087.   , 16140006.   , 16141970.   , 16143800.   ,
        16145690.   , 16147688.   , 16149473.   , 16151409.   ,
        16153328.   , 16155188.   , 16157138.   , 16159050.   ],
       [16669960.   , 16672369.   , 16674345.   , 16676367.   ,
        16678274.   , 16680258.   , 16682287.   , 16684178.   ,
        16686132.   , 16688196.   , 16690041.   , 16692042.   ,
        16694026.   , 16695948.   , 16697962.   , 16699938.   ],
       [17209878.   , 17212364.   , 17214404.   , 17216492.   ,
        17218460.   , 17220508.   , 17222604.   , 17224556.   ,
        17226572.   , 17228704.   , 17230608.   , 17232676.   ,
        17234724.   , 17236708.   , 17238788.   , 17240828.   ],
       [17817286.   , 17819860.   , 17821972.   , 17824132.   ,
        17826172.   , 17828292.   , 17830460.   , 17832482.   ,
        17834570.   , 17836776.   , 17838748.   , 17840888.   ,
        17843008.   , 17845062.   , 17847216.   , 17849328.   ],
       [18357204.   , 18359856.   , 18362032.   , 18364258.   ,
        18366358.   , 18368542.   , 18370778.   , 18372860.   ,
        18375012.   , 18377284.   , 18379316.   , 18381520.   ,
        18383704.   , 18385820.   , 18388040.   , 18390216.   ],
       [18897120.   , 18899852.   , 18902092.   , 18904384.   ,
        18906544.   , 18908794.   , 18911096.   , 18913240.   ,
        18915452.   , 18917792.   , 18919884.   , 18922152.   ,
        18924402.   , 18926580.   , 18928864.   , 18931104.   ],
       [19437040.   , 19439848.   , 19442152.   , 19444508.   ,
        19446732.   , 19449044.   , 19451412.   , 19453616.   ,
        19455894.   , 19458302.   , 19460452.   , 19462786.   ,
        19465100.   , 19467340.   , 19469688.   , 19471992.   ],
       [19976956.   , 19979844.   , 19982212.   , 19984634.   ,
        19986920.   , 19989296.   , 19991728.   , 19993996.   ,
        19996336.   , 19998810.   , 20001020.   , 20003420.   ,
        20005796.   , 20008100.   , 20010514.   , 20012882.   ],
       [20516874.   , 20519838.   , 20522270.   , 20524760.   ,
        20527106.   , 20529548.   , 20532046.   , 20534374.   ,
        20536776.   , 20539318.   , 20541588.   , 20544052.   ,
        20546492.   , 20548858.   , 20551338.   , 20553770.   ],
       [21056792.   , 21059834.   , 21062330.   , 21064884.   ,
        21067292.   , 21069800.   , 21072364.   , 21074752.   ,
        21077218.   , 21079826.   , 21082156.   , 21084684.   ,
        21087190.   , 21089618.   , 21092164.   , 21094660.   ],
       [21596710.   , 21599830.   , 21602390.   , 21605010.   ,
        21607480.   , 21610050.   , 21612680.   , 21615130.   ,
        21617660.   , 21620336.   , 21622724.   , 21625318.   ,
        21627888.   , 21630378.   , 21632988.   , 21635548.   ],
       [22218698.   , 22221906.   , 22224536.   , 22227228.   ,
        22229768.   , 22232408.   , 22235108.   , 22237628.   ,
        22240228.   , 22242976.   , 22245432.   , 22248094.   ,
        22250736.   , 22253292.   , 22255972.   , 22258602.   ],
       [22802946.   , 22806238.   , 22808938.   , 22811700.   ,
        22814306.   , 22817016.   , 22819790.   , 22822374.   ,
        22825044.   , 22827864.   , 22830384.   , 22833120.   ,
        22835830.   , 22838456.   , 22841208.   , 22843908.   ],
       [23351442.   , 23354816.   , 23357584.   , 23360416.   ,
        23363088.   , 23365866.   , 23368710.   , 23371360.   ,
        23374096.   , 23376988.   , 23379572.   , 23382374.   ,
        23385154.   , 23387846.   , 23390668.   , 23393436.   ],
       [23891360.   , 23894812.   , 23897644.   , 23900542.   ,
        23903276.   , 23906118.   , 23909028.   , 23911738.   ,
        23914536.   , 23917496.   , 23920140.   , 23923008.   ,
        23925850.   , 23928604.   , 23931492.   , 23934324.   ],
       [24431278.   , 24434808.   , 24437704.   , 24440668.   ,
        24443462.   , 24446368.   , 24449344.   , 24452116.   ,
        24454978.   , 24458004.   , 24460708.   , 24463640.   ,
        24466548.   , 24469364.   , 24472316.   , 24475212.   ],
       [24971196.   , 24974804.   , 24977764.   , 24980792.   ,
        24983648.   , 24986620.   , 24989662.   , 24992494.   ,
        24995420.   , 24998512.   , 25001276.   , 25004274.   ,
        25007244.   , 25010124.   , 25013142.   , 25016102.   ],
       [25511114.   , 25514800.   , 25517824.   , 25520918.   ,
        25523836.   , 25526872.   , 25529978.   , 25532872.   ,
        25535860.   , 25539020.   , 25541844.   , 25544906.   ,
        25547942.   , 25550884.   , 25553966.   , 25556990.   ],
       [26051032.   , 26054796.   , 26057884.   , 26061044.   ,
        26064024.   , 26067124.   , 26070296.   , 26073250.   ,
        26076302.   , 26079528.   , 26082412.   , 26085540.   ,
        26088640.   , 26091644.   , 26094792.   , 26097880.   ],
       [26590950.   , 26594792.   , 26597944.   , 26601168.   ,
        26604210.   , 26607374.   , 26610612.   , 26613628.   ,
        26616744.   , 26620038.   , 26622980.   , 26626172.   ,
        26629336.   , 26632402.   , 26635616.   , 26638768.   ],
       [27130868.   , 27134786.   , 27138002.   , 27141294.   ,
        27144396.   , 27147624.   , 27150930.   , 27154008.   ,
        27157184.   , 27160546.   , 27163548.   , 27166804.   ,
        27170034.   , 27173162.   , 27176440.   , 27179656.   ],
       [27723244.   , 27727248.   , 27730532.   , 27733892.   ,
        27737062.   , 27740358.   , 27743732.   , 27746876.   ,
        27750120.   , 27753552.   , 27756618.   , 27759944.   ,
        27763240.   , 27766436.   , 27769780.   , 27773064.   ],
       [28323220.   , 28327310.   , 28330664.   , 28334094.   ,
        28337332.   , 28340696.   , 28344142.   , 28347352.   ,
        28350664.   , 28354168.   , 28357300.   , 28360696.   ,
        28364062.   , 28367324.   , 28370744.   , 28374096.   ],
       [28885444.   , 28889616.   , 28893040.   , 28896544.   ,
        28899848.   , 28903284.   , 28906802.   , 28910078.   ,
        28913460.   , 28917038.   , 28920236.   , 28923702.   ,
        28927138.   , 28930468.   , 28933960.   , 28937382.   ],
       [29425518.   , 29429768.   , 29433256.   , 29436826.   ,
        29440192.   , 29443692.   , 29447276.   , 29450614.   ,
        29454062.   , 29457706.   , 29460964.   , 29464496.   ,
        29467996.   , 29471390.   , 29474946.   , 29478434.   ],
       [29965436.   , 29969764.   , 29973316.   , 29976952.   ,
        29980378.   , 29983944.   , 29987594.   , 29990992.   ,
        29994504.   , 29998216.   , 30001532.   , 30005128.   ,
        30008694.   , 30012148.   , 30015770.   , 30019322.   ],
       [30505352.   , 30509760.   , 30513376.   , 30517076.   ,
        30520566.   , 30524196.   , 30527910.   , 30531372.   ,
        30534944.   , 30538724.   , 30542100.   , 30545760.   ,
        30549392.   , 30552908.   , 30556596.   , 30560212.   ],
       [31045270.   , 31049756.   , 31053436.   , 31057202.   ,
        31060752.   , 31064448.   , 31068228.   , 31071750.   ,
        31075386.   , 31079232.   , 31082668.   , 31086394.   ,
        31090088.   , 31093668.   , 31097420.   , 31101100.   ],
       [31585188.   , 31589752.   , 31593496.   , 31597328.   ,
        31600940.   , 31604698.   , 31608544.   , 31612128.   ,
        31615828.   , 31619740.   , 31623236.   , 31627028.   ,
        31630786.   , 31634428.   , 31638244.   , 31641988.   ],
       [32125106.   , 32129748.   , 32133556.   , 32137452.   ,
        32141126.   , 32144950.   , 32148862.   , 32152506.   ,
        32156268.   , 32160248.   , 32163804.   , 32167660.   ,
        32171482.   , 32175186.   , 32179068.   , 32182876.   ],
       [32665024.   , 32669744.   , 32673616.   , 32677578.   ,
        32681314.   , 32685200.   , 32689178.   , 32692884.   ,
        32696712.   , 32700756.   , 32704372.   , 32708292.   ,
        32712180.   , 32715946.   , 32719894.   , 32723766.   ],
       [33221238.   , 33226038.   , 33229974.   , 33234004.   ,
        33237804.   , 33241756.   , 33245802.   , 33249570.   ,
        33253460.   , 33257576.   , 33261252.   , 33265238.   ,
        33269192.   , 33273022.   , 33277034.   , 33280972.   ],
       [33836944.   , 33841824.   , 33845832.   , 33849936.   ,
        33853804.   , 33857824.   , 33861940.   , 33865776.   ,
        33869736.   , 33873920.   , 33877664.   , 33881720.   ,
        33885744.   , 33889640.   , 33893724.   , 33897732.   ],
       [34414896.   , 34419864.   , 34423944.   , 34428112.   ,
        34432048.   , 34436140.   , 34440328.   , 34444232.   ,
        34448260.   , 34452520.   , 34456324.   , 34460456.   ,
        34464548.   , 34468512.   , 34472672.   , 34476748.   ],
       [34824696.   , 34829728.   , 34833856.   , 34838080.   ,
        34842064.   , 34846208.   , 34850448.   , 34854396.   ,
        34858476.   , 34862792.   , 34866644.   , 34870824.   ,
        34874968.   , 34878984.   , 34883192.   , 34887320.   ]],
      dtype=float32),),
    mlir_module_text=r"""
#loc6 = loc("third_party/py/jax/tests/pallas/export_back_compat_pallas_test.py":74:12)
#loc14 = loc("jit(func)/jit(main)/pjit"(#loc6))
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<64x16xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %c_0 = stablehlo.constant dense<16> : tensor<i32> loc(#loc)
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32> loc(#loc)
    %cst_1 = stablehlo.constant dense<1.000000e-03> : tensor<f32> loc(#loc)
    %0 = stablehlo.iota dim = 0 : tensor<524288xf32> loc(#loc9)
    %1 = stablehlo.reshape %0 : (tensor<524288xf32>) -> tensor<1024x512xf32> loc(#loc10)
    %2 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x512xf32> loc(#loc11)
    %3 = stablehlo.multiply %2, %1 : tensor<1024x512xf32> loc(#loc11)
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x512xf32> loc(#loc12)
    %5 = stablehlo.add %4, %3 : tensor<1024x512xf32> loc(#loc12)
    %6 = stablehlo.slice %5 [0:512, 0:256] : (tensor<1024x512xf32>) -> tensor<512x256xf32> loc(#loc13)
    %7 = call @matmul(%5, %6) : (tensor<1024x512xf32>, tensor<512x256xf32>) -> tensor<1024x256xf32> loc(#loc14)
    %8 = stablehlo.iota dim = 0 : tensor<64xi32> loc(#loc15)
    %9 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<64xi32> loc(#loc16)
    %10 = stablehlo.multiply %9, %8 : tensor<64xi32> loc(#loc16)
    %11 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<64xi32> loc(#loc17)
    %12 = stablehlo.add %11, %10 : tensor<64xi32> loc(#loc17)
    %13 = stablehlo.iota dim = 0 : tensor<16xi32> loc(#loc15)
    %14 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<16xi32> loc(#loc16)
    %15 = stablehlo.multiply %14, %13 : tensor<16xi32> loc(#loc16)
    %16 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<16xi32> loc(#loc17)
    %17 = stablehlo.add %16, %15 : tensor<16xi32> loc(#loc17)
    %18 = stablehlo.broadcast_in_dim %12, dims = [0] : (tensor<64xi32>) -> tensor<64x16x1xi32> loc(#loc18)
    %19 = stablehlo.broadcast_in_dim %17, dims = [1] : (tensor<16xi32>) -> tensor<64x16x1xi32> loc(#loc18)
    %20 = stablehlo.concatenate %18, %19, dim = 2 : (tensor<64x16x1xi32>, tensor<64x16x1xi32>) -> tensor<64x16x2xi32> loc(#loc19)
    %21 = "stablehlo.gather"(%7, %20) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 2>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1>}> : (tensor<1024x256xf32>, tensor<64x16x2xi32>) -> tensor<64x16xf32> loc(#loc20)
    return %21 : tensor<64x16xf32> loc(#loc)
  } loc(#loc)
  func.func private @matmul(%arg0: tensor<1024x512xf32> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc6)), %arg1: tensor<512x256xf32> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc6))) -> (tensor<1024x256xf32> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.custom_call @tpu_custom_call(%arg0, %arg1) {backend_config = "{\22custom_call_config\22: {\22body\22: \22TUzvUgFNTElSZ29vZ2xlMy10cnVuawABLQkBAwUHAQMJAxkLDQ8RExUXGRsdHyED58ETAbkHEwsTCwsLDwsPDw8LC1MLDw8PDwsPDw8LCwsLExMPDxMPGwsPC0MLFwuFC3MLCwsLFxsLGwsbCxsbGw8LExMPEw8LCxMPExMTHwsTGwsLEwsPCxMTEwsTDwsTEwUHjZFhBwNZARMPBx8nDwcLKyMCZggfAwMLiwUjAwMLdwUlBScFKR15ewUrHSmnHSmrHSm3BS0FLyMJBSEAAQAAAAAAAAABAAAAAAAADREdhzkdETsdEY0dEY8FMR0RqREJAREJBQUzBTUFNwU5FwU7BxcFQyMdlZcRDQAXrRcLHbO1AwVHSQlLBTsRCQ0FPQMPT1ENU1dZWy1dLwlfYWMFPwEHubm7DQ9hZmZpbmVfbWFwPChkMCwgZDEpIC0+IChkMCwgZDEpPgAFQSMJBzEEAAAAAAAAAAEAAAAAAAAAAgAAAAAAAAAFQwVFBUcFSQEHZWltAwUZZxsdCTEDBRlrGx0JMwMFGW8bHQk1AwUNHwkxAwUNHwkzAwUNHwk1EQEBBUsXBTsXAwMLfxEBBQMDNy0dhTkFTQVPAwM3LxEDARcFRQ0XBUcNAwMLkyUFCQAAAAAFURcFQ0EDBZs/nT8FUwVVAwOhvwVXHaU7BVkXBUMFFwVRKRcFUQUFWwMDC7ETCwEFXRcFPycXBT8JI3RwdS5kaW1lbnNpb25fc2VtYW50aWNzPHBhcmFsbGVsPgAjdHB1LmRpbWVuc2lvbl9zZW1hbnRpY3M8YXJiaXRyYXJ5PgAjdHB1Lm1lbW9yeV9zcGFjZTx2bWVtPgAjYXJpdGguZmFzdG1hdGg8bm9uZT4AAQICAycFAggCCAsXvQUCCAIIC1UBAgQLAQkFDwEBAQcHBwcBBQcBAQEFAQEEIgYFAREBRQcDAREHEQFNBwNHgw8BAQEBAQEHAQcBBwEHAQMDDwcDAQMDD30DAQMDDwcDAQ0HD4EDDQUFDxEGgwMBAxUDAyEHAwENByGJAw0FFxkTFCEDGwkDCx0DA0OvAwsZBkMDBQNHAwMXAwMDAwMXAwMDBQYXAwUHDUtNCwQXCUkNS00PAEEDAQUPAEEDAyMDAwMDAyMDAwMFBiMDBQcNHR8DAyUDAwMDAyUDAwMFBiUDBQcHIyUDAycDAwMDAycDAwMFBicDBQcJKSsDAz2RAwUVBz2ZAwUHJy0vFwejnwMFBSExAwMTAwMDAwMTAwMDBQYTAwUHDTU3CwQTCTMNNTcDAysDAwMDAysDAwMFBisDBQcNOz0DAxUDAwMDAxUDAwMFBhUDBQcLQUMLBBUJPwtBQwkAAQcRAXEHAwkLBwEBAQEBAQMDAQcDAQkEAQUBBQcRAXMHAwkLBwEBAQEBAQMDAQcDAQkEAQUFAwcRAXUHAwkLBwEBAQEBAQMDAQcDAQkEAQUBAwYDAQUBAO4JXyUFCxMdHRsNLQkdCyMhIykdLRUZGRkNHSULHQ0TcyMXFw8ZFRcbGRUZHw8NCR0RYnVpbHRpbgBzdGFibGVfbW9zYWljAHRwdQBhcml0aABtb2R1bGUAYXJpdGguY29uc3RhbnQAdmVjdG9yLmxvYWQAZnVuYy5mdW5jAGZ1bmMucmV0dXJuAHZlY3Rvci5zdG9yZQBhcml0aC5jbXBpAHNjZi55aWVsZABhcml0aC5leHR1aQBzY2YuaWYAdHB1Lm1hdG11bABhcml0aC5hZGRmAHZlY3Rvci5icm9hZGNhc3QAdGhpcmRfcGFydHkvcHkvamF4L2V4cGVyaW1lbnRhbC9wYWxsYXMvb3BzL3RwdS9tYXRtdWwucHkAc3ltX25hbWUAdmFsdWUAZnVuY3Rpb25fdHlwZQAvZ2V0AHRyYW5zZm9ybV9pbmRpY2VzAHdpbmRvd19ib3VuZHMAL3N3YXAAdHJhbnNmb3JtXzAAdHJhbnNmb3JtXzEAdHJhbnNmb3JtXzIAcHJlZGljYXRlAHN0YWJsZV9tb3NhaWMudmVyc2lvbgBtYXRtdWxfa2VybmVsAGRpbWVuc2lvbl9zZW1hbnRpY3MAaXRlcmF0aW9uX2JvdW5kcwBzY2FsYXJfcHJlZmV0Y2gAc2NyYXRjaF9vcGVyYW5kcwBtYWluAHdpbmRvd19wYXJhbXMAL2VxAC9jb252ZXJ0X2VsZW1lbnRfdHlwZQAvY29uZAAvZG90X2dlbmVyYWwAdHJhbnNwb3NlX2xocwB0cmFuc3Bvc2VfcmhzAGZhc3RtYXRoAC9hZGQALQAvYnJvYWRjYXN0X2luX2RpbQA=\22, \22serialization_format\22: 1, \22needs_layout_passes\22: true}, \22implicit_sharding\22: {\22type\22: \22MANUAL\22}}", kernel_name = "matmul_kernel", operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>], result_layouts = [dense<[1, 0]> : tensor<2xindex>]} : (tensor<1024x512xf32>, tensor<512x256xf32>) -> tensor<1024x256xf32> loc(#loc21)
    return %0 : tensor<1024x256xf32> loc(#loc14)
  } loc(#loc14)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/pallas/export_back_compat_pallas_test.py":71:25)
#loc2 = loc("third_party/py/jax/tests/pallas/export_back_compat_pallas_test.py":72:43)
#loc3 = loc("third_party/py/jax/tests/pallas/export_back_compat_pallas_test.py":71:17)
#loc4 = loc("third_party/py/jax/tests/pallas/export_back_compat_pallas_test.py":71:10)
#loc5 = loc("third_party/py/jax/tests/pallas/export_back_compat_pallas_test.py":73:10)
#loc7 = loc("third_party/py/jax/tests/pallas/export_back_compat_pallas_test.py":76:13)
#loc8 = loc("third_party/py/jax/experimental/pallas/ops/tpu/matmul.py":68:9)
#loc9 = loc("jit(func)/jit(main)/iota"(#loc1))
#loc10 = loc("jit(func)/jit(main)/reshape"(#loc2))
#loc11 = loc("jit(func)/jit(main)/mul"(#loc3))
#loc12 = loc("jit(func)/jit(main)/add"(#loc4))
#loc13 = loc("jit(func)/jit(main)/slice"(#loc5))
#loc15 = loc("jit(func)/jit(main)/iota"(#loc7))
#loc16 = loc("jit(func)/jit(main)/mul"(#loc7))
#loc17 = loc("jit(func)/jit(main)/add"(#loc7))
#loc18 = loc("jit(func)/jit(main)/broadcast_in_dim"(#loc7))
#loc19 = loc("jit(func)/jit(main)/concatenate"(#loc7))
#loc20 = loc("jit(func)/jit(main)/gather"(#loc7))
#loc21 = loc("jit(func)/jit(main)/jit(matmul)/pallas_call"(#loc8))
""",
    mlir_module_serialized=b'ML\xefR\rStableHLO_v1.5.0\x00\x01-\x05\x01\x05\x1d\x01\x03\x0b\x03\x1b\x0f\x13\x17\x1b\x1f#\'+/37;?\x03\xe5\xa3/\x01W\x07\x0b\x13\x0f\x0f\x0f\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0b\x13\x13\x0b\x0f\x0b\x13\x0b\x0f\x13\x0f\x0b\x13\x13\x13\x0f\x0b\x13\x0b\x0f\x0b\x0f\x0b\x03M\x0f\x0b\x13O\x0f\x0b\x0b\x0b/\x0fO\x0b\x0f\x1b\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x13\x0f\x1f\x1f\x1f\x1fO///\x0b\x01\x05\x0b\x0f\x03+\x1f\x07\x07\x07\x17\x13\x0f\x0f\x13\x1f\x1f\x1b\x1f\x13\x13\x1b\x13\x07\x1b\x13\x1f\x02\x92\x06\x1f\x05!\x17\x03\x99\x1b\x1d)+\x1d\x13\x05\x1d\x17\x05\x11\x03\x05\x05#\x1d\x13C\x05%\x1d\x17E\x05\'\x1d\x0f\x05\x1dM\x05\x03\x07\x1f!#\r%\r\x05)\x11\x01\x00\x05+\x05-\x05/\x051\x17\x03\x95\x19\x03\x03/\x83\x053\x1d35\x055\x177\x89\x13\x057\x1d\x0f;\x17\x03\x8f3\x1d?A\x059\x17\x03\x91W\x17\x03\x8f#\x17\x03\x8f\x15\x1dIK\x05;\x17\x03\x93\x15\x05=\x1dQ\x05\x05?\x1dU\x05\x05A\x1f+\x01\x03\x01\r\x03ac\x1f%!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x13\x0b\x01\x1dC\x1dE\x1dG\x1f\x15\x11\x01\x00\x00\x00\x00\x00\x00\x00\x13\x0b\t\x1f\x15!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00#!\x03\x03q\r\x05suac\x1dI\x1dK\x1dM\x1dO\x03\x05[[##\x03\x03[\x1dQ\x1dS\x0b\x03\x1dU\x1dW\x05\x01\x03\x05]]\x03\x03]\x1f\x11\t\x00\x00\x00\x00\x1f\x11\t\x10\x00\x00\x00\x1f\x13\t\x00\x00\x80?\x1f\x13\to\x12\x83:\x1f\x15!\x00\x02\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x1f\x15\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x1f\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x1f\x11\x01\x00\x00\x00\x00\x00\x00\x00\x05\x03\x01\t\x01\x02\x02)\x05\x02 \x02\x10\x07\t\x1b\x1d)\x03\x02\x02\t)\x03A\t)\x01\t)\x01\x07)\x03\t\x0b)\x05\x02\x10\x02\x08\x07)\x05\x02 \x02\x08\x07)\x05\x02\x02A\x07)\x07\x02\x02A\x05\t)\x03\x05\x0b\x11\x01\x03\x1b\x11\x05\x05\x17\x03\x19)\x03\t\'\x13)\x03\x04\x00\x80\x07)\x03\x01\x0b)\x07\x02\x02A\t\t\x04\x02\x04\x05\x01Q\x01\x1d\x01\x07\x04\xda\x03\x03\x01\t\rP\x01\x03\x07\x042\x03\x035m\x05B\x01\x05\x03\x11\x05B\x01\x07\x03\x11\x05B\x01\t\x03\x13\x05B\x01\x0b\x03\x13\x07B9\r\x03)\x13\x06=\x03\x05\x03\t\x03F\x11\x0f\x03\x05\x03\x07\t\x06\x11\x03\x05\x05\r\x0b\x03F\x15\x0f\x03\x05\x03\x05\x0b\x06\x15\x03\x05\x05\x11\x0f\x15FG\x11\x03\x17\x03\x13\x17F\x07\x13\x03\x19\x05\x13\x15\x07B\x19\r\x03\r\x03F\t\x0f\x03\r\x03\x03\t\x06\t\x03\r\x05\x1b\x19\x03F\x0b\x0f\x03\r\x03\x01\x0b\x06\x0b\x03\r\x05\x1f\x1d\x07B\x19\r\x03\x0f\x03F\t\x0f\x03\x0f\x03\x03\t\x06\t\x03\x0f\x05%#\x03F\x0b\x0f\x03\x0f\x03\x01\x0b\x06\x0b\x03\x0f\x05)\'\x03F\x1b\x15\x03\x1d\x03!\x03F\x1b\x17\x03\x1d\x03+\x19FO\x19\x03-\x05-/\x1bFS\x1b\x03\x1b\x05\x171\x0f\x04\x01\x033\rP\x07\x1d\x07\x041\x03\x07\x0b\x05\x0b\x07/\x07\x00\x11G1-\x1f\x03\x19\x05\x01\x03\x0f\x04\x07\x03\x05\x06\x03\x01\x05\x01\x00\x1a3Y!j&\x1d\x11\x0f\x0b\x03!\x0f\x11#7AK59sY\x193\x13%)9113\x85\x15\x1f\x11\x13\x17\x1f\x15\x11\x0f\x19\x11\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00iota_v1\x00multiply_v1\x00add_v1\x00func_v1\x00return_v1\x00custom_call_v1\x00reshape_v1\x00slice_v1\x00call_v1\x00concatenate_v1\x00gather_v2\x00third_party/py/jax/tests/pallas/export_back_compat_pallas_test.py\x00jit(func)/jit(main)/iota\x00jit(func)/jit(main)/mul\x00jit(func)/jit(main)/add\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00jit(func)/jit(main)/pjit\x00kernel_name\x00jit(func)/jit(main)/jit(matmul)/pallas_call\x00third_party/py/jax/experimental/pallas/ops/tpu/matmul.py\x00jit(func)/jit(main)/reshape\x00jit(func)/jit(main)/slice\x00jit(func)/jit(main)/broadcast_in_dim\x00jit(func)/jit(main)/concatenate\x00jit(func)/jit(main)/gather\x00mhlo.layout_mode\x00default\x00matmul\x00jax.result_info\x00\x00main\x00public\x00private\x00matmul_kernel\x00{"custom_call_config": {"body": "TUzvUgFNTElSZ29vZ2xlMy10cnVuawABLQkBAwUHAQMJAxkLDQ8RExUXGRsdHyED58ETAbkHEwsTCwsLDwsPDw8LC1MLDw8PDwsPDw8LCwsLExMPDxMPGwsPC0MLFwuFC3MLCwsLFxsLGwsbCxsbGw8LExMPEw8LCxMPExMTHwsTGwsLEwsPCxMTEwsTDwsTEwUHjZFhBwNZARMPBx8nDwcLKyMCZggfAwMLiwUjAwMLdwUlBScFKR15ewUrHSmnHSmrHSm3BS0FLyMJBSEAAQAAAAAAAAABAAAAAAAADREdhzkdETsdEY0dEY8FMR0RqREJAREJBQUzBTUFNwU5FwU7BxcFQyMdlZcRDQAXrRcLHbO1AwVHSQlLBTsRCQ0FPQMPT1ENU1dZWy1dLwlfYWMFPwEHubm7DQ9hZmZpbmVfbWFwPChkMCwgZDEpIC0+IChkMCwgZDEpPgAFQSMJBzEEAAAAAAAAAAEAAAAAAAAAAgAAAAAAAAAFQwVFBUcFSQEHZWltAwUZZxsdCTEDBRlrGx0JMwMFGW8bHQk1AwUNHwkxAwUNHwkzAwUNHwk1EQEBBUsXBTsXAwMLfxEBBQMDNy0dhTkFTQVPAwM3LxEDARcFRQ0XBUcNAwMLkyUFCQAAAAAFURcFQ0EDBZs/nT8FUwVVAwOhvwVXHaU7BVkXBUMFFwVRKRcFUQUFWwMDC7ETCwEFXRcFPycXBT8JI3RwdS5kaW1lbnNpb25fc2VtYW50aWNzPHBhcmFsbGVsPgAjdHB1LmRpbWVuc2lvbl9zZW1hbnRpY3M8YXJiaXRyYXJ5PgAjdHB1Lm1lbW9yeV9zcGFjZTx2bWVtPgAjYXJpdGguZmFzdG1hdGg8bm9uZT4AAQICAycFAggCCAsXvQUCCAIIC1UBAgQLAQkFDwEBAQcHBwcBBQcBAQEFAQEEIgYFAREBRQcDAREHEQFNBwNHgw8BAQEBAQEHAQcBBwEHAQMDDwcDAQMDD30DAQMDDwcDAQ0HD4EDDQUFDxEGgwMBAxUDAyEHAwENByGJAw0FFxkTFCEDGwkDCx0DA0OvAwsZBkMDBQNHAwMXAwMDAwMXAwMDBQYXAwUHDUtNCwQXCUkNS00PAEEDAQUPAEEDAyMDAwMDAyMDAwMFBiMDBQcNHR8DAyUDAwMDAyUDAwMFBiUDBQcHIyUDAycDAwMDAycDAwMFBicDBQcJKSsDAz2RAwUVBz2ZAwUHJy0vFwejnwMFBSExAwMTAwMDAwMTAwMDBQYTAwUHDTU3CwQTCTMNNTcDAysDAwMDAysDAwMFBisDBQcNOz0DAxUDAwMDAxUDAwMFBhUDBQcLQUMLBBUJPwtBQwkAAQcRAXEHAwkLBwEBAQEBAQMDAQcDAQkEAQUBBQcRAXMHAwkLBwEBAQEBAQMDAQcDAQkEAQUFAwcRAXUHAwkLBwEBAQEBAQMDAQcDAQkEAQUBAwYDAQUBAO4JXyUFCxMdHRsNLQkdCyMhIykdLRUZGRkNHSULHQ0TcyMXFw8ZFRcbGRUZHw8NCR0RYnVpbHRpbgBzdGFibGVfbW9zYWljAHRwdQBhcml0aABtb2R1bGUAYXJpdGguY29uc3RhbnQAdmVjdG9yLmxvYWQAZnVuYy5mdW5jAGZ1bmMucmV0dXJuAHZlY3Rvci5zdG9yZQBhcml0aC5jbXBpAHNjZi55aWVsZABhcml0aC5leHR1aQBzY2YuaWYAdHB1Lm1hdG11bABhcml0aC5hZGRmAHZlY3Rvci5icm9hZGNhc3QAdGhpcmRfcGFydHkvcHkvamF4L2V4cGVyaW1lbnRhbC9wYWxsYXMvb3BzL3RwdS9tYXRtdWwucHkAc3ltX25hbWUAdmFsdWUAZnVuY3Rpb25fdHlwZQAvZ2V0AHRyYW5zZm9ybV9pbmRpY2VzAHdpbmRvd19ib3VuZHMAL3N3YXAAdHJhbnNmb3JtXzAAdHJhbnNmb3JtXzEAdHJhbnNmb3JtXzIAcHJlZGljYXRlAHN0YWJsZV9tb3NhaWMudmVyc2lvbgBtYXRtdWxfa2VybmVsAGRpbWVuc2lvbl9zZW1hbnRpY3MAaXRlcmF0aW9uX2JvdW5kcwBzY2FsYXJfcHJlZmV0Y2gAc2NyYXRjaF9vcGVyYW5kcwBtYWluAHdpbmRvd19wYXJhbXMAL2VxAC9jb252ZXJ0X2VsZW1lbnRfdHlwZQAvY29uZAAvZG90X2dlbmVyYWwAdHJhbnNwb3NlX2xocwB0cmFuc3Bvc2VfcmhzAGZhc3RtYXRoAC9hZGQALQAvYnJvYWRjYXN0X2luX2RpbQA=", "serialization_format": 1, "needs_layout_passes": true}, "implicit_sharding": {"type": "MANUAL"}}\x00tpu_custom_call\x00\x08u!\x05O\x01\x0bYmowy\x03\x91\x03\x93\x03\x95\x03\x97\x03_\x03W\x07\x99\x9bg\x03e\x03\x9d\x03\x9f\x03i\x11ki\xa1WWgkW\x0b{}\x7fe\x81\x11\x85\x87\x89Y\x8b\x8dY\x8f',
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste
