"""Physical task ordering. Motion implementation lives in lab_motion / MuJoCoBackend."""
def build_physical_flow(state,backend,settings,action,sequence,finish):
    def a(name,kind=None):
        return action(name,**({'fault_on_failure':kind} if kind else {}))
    source=sequence('SourceToAluminium',
        a('InitializeMission'),a('SelectSourcePlate'),
        a('SourceHigh'),a('SourceApproach'),a('SourceDescend'),a('SourceClose','grasp'),
        a('SourceLift'),a('VerifySourceGrasp','grasp'),a('SourceHighTransit'),a('AluminiumClearTransit'),
        a('AluminiumApproachForPlace'),a('AluminiumDescendForPlace'),a('AluminiumRelease'),
        a('AluminiumRetractAfterPlace'),a('VerifyAluminiumPlaced','placement'),a('HomeAfterAluminium'))
    reader=sequence('AluminiumToReader',
        a('AluminiumApproachForPick'),a('AluminiumDescendForPick'),a('AluminiumClose','grasp'),
        a('AluminiumLift'),a('VerifyAluminiumGrasp','grasp'),a('AluminiumExitTransit'),a('HomeBeforeReader'),
        a('ReaderApproachForPlace'),a('AlignWithReader','alignment'),a('VerifyReaderAligned','alignment'),
        a('ReaderDescendForPlace'),a('LoadPlateIntoReader','alignment'),a('ReaderRetractAfterPlace'),
        a('VerifyReaderLoaded','placement'),a('HomeAfterReader'),a('StartReaderMeasurement'),a('WaitForReadingComplete'))
    storage=sequence('ReaderToSlot2',
        a('ReaderApproachForPick'),a('ReaderDescendForPick'),a('ReaderClose','grasp'),
        a('ReaderLift'),a('VerifyReaderGrasp','grasp'),a('HomeBeforeStorage'),
        a('ReturnToStorageSlot'),a('AlignWithStorageSlot','alignment'),a('VerifyStorageAligned','alignment'),
        a('StorageDescend'),a('PlacePlateInSlot','placement'),a('StorageRetract'),a('VerifyReturned','placement'),
        a('HomeAfterStorage'),a('CommitStoredPlate'),finish(state,outcome='measured'))
    return sequence('04_PHYSICAL_ONE_PLATE',source,reader,storage)
