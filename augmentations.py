
from transforms import RandomCutoutNp
from transforms import SpecAugmentNp
from transforms import RandomCutoutHoleNp
from transforms import RandomShiftUpDownNp




cutter=RandomCutoutNp()
shifter=RandomShiftUpDownNp()
driller=RandomCutoutHoleNp()
specaug=SpecAugmentNp()

augmentations = [
cutter, shifter, driller, specaug
]