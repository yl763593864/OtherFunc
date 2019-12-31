/*
 Navicat MongoDB Data Transfer

 Source Server         : localhost_27017
 Source Server Type    : MongoDB
 Source Server Version : 40200
 Source Host           : localhost:27017
 Source Schema         : english

 Target Server Type    : MongoDB
 Target Server Version : 40200
 File Encoding         : 65001

 Date: 31/12/2019 20:18:56
*/


// ----------------------------
// Collection structure for english_words
// ----------------------------
db.getCollection("english_words").drop();
db.createCollection("english_words");

// ----------------------------
// Documents of english_words
// ----------------------------
db.getCollection("english_words").insert([ {
    _id: ObjectId("5d8433cc72a3b81b52cbbf02"),
    word: "xxs",
    translation: "",
    phrase: "",
    "example_sentence": "",
    synonym: "",
    "create_time": ISODate("2019-09-20T10:05:00.228Z")
} ]);
db.getCollection("english_words").insert([ {
    _id: ObjectId("5d8434c6222a0b48a222b455"),
    word: "xxse",
    translation: "",
    phrase: "",
    "example_sentence": "",
    synonym: "",
    "create_time": ISODate("2019-09-20T10:09:10.594Z")
} ]);

// ----------------------------
// Collection structure for words
// ----------------------------
db.getCollection("words").drop();
db.createCollection("words");

// ----------------------------
// Documents of words
// ----------------------------
db.getCollection("words").insert([ {
    _id: ObjectId("5e01dfe5f5756063082a0d48"),
    word: "presumably",
    sentence: [
        "presumably this is where the accident happened.",
        "You'll be taking the car, Presumably.",
        "I couldn't concentrate, presumably because I was so tired.",
        "I couldn't concentrate, presumably because I was so tired."
    ],
    similar: [ ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e01dfe5f5756063082a0d49"),
    word: "assist",
    sentence: [
        "Anyone willing to assist can contract this number."
    ],
    similar: [
        "asset"
    ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e01dfe5f5756063082a0d4a"),
    word: "decaying",
    sentence: [
        "decaying inner city areas."
    ],
    similar: [ ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e01dfe5f5756063082a0d4b"),
    word: "reserve",
    sentence: [
        "large oil and gas reserve"
    ],
    similar: [
        "revenge"
    ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e01dfe5f5756063082a0d4c"),
    word: "guarantee",
    sentence: [
        "Basic human rights, including freedom of speech, are now guaranteed"
    ],
    similar: [ ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e01dfe5f5756063082a0d4d"),
    word: "delicate",
    sentence: [
        "delicate china teacups",
        "The eye is one of the most delicate organs of the body."
    ],
    similar: [ ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e01dfe5f5756063082a0d4e"),
    word: "petrol",
    sentence: [
        "to fill a car up with petrol",
        "to run out of petrol",
        "an increase in petrol price"
    ],
    similar: [ ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e01dfe5f5756063082a0d4f"),
    word: "sufficient",
    sentence: [
        "Alow sufficient time to get there."
    ],
    similar: [ ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e01dfe5f5756063082a0d50"),
    word: "bite",
    sentence: [
        "Dose your dog bite?",
        "She bit into a ripe juicy pear."
    ],
    similar: [
        "byte",
        "bit"
    ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e01dfe5f5756063082a0d51"),
    word: "rival",
    sentence: [
        "The two teams have always been rivals",
        "The japanese are our biggest economic rivals."
    ],
    similar: [
        "competitor"
    ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e01dfe5f5756063082a0d52"),
    word: "mention",
    sentence: [
        "Nobody mentioned anything to me about it.",
        "Sorry, I won't mention it again."
    ],
    similar: [ ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e01dfe5f5756063082a0d53"),
    word: "opposition",
    sentence: [
        "Delegates expressed strong opposition to the plan"
    ],
    similar: [ ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e01dfe5f5756063082a0d54"),
    word: "ambition",
    sentence: [
        "It had been her lifelong ambition.",
        "political/literary/sporting ambition"
    ],
    similar: [ ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e01dfe5f5756063082a0d55"),
    word: "clerk",
    sentence: [
        "an office clerk",
        "an Town clerk"
    ],
    similar: [ ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e01dfe5f5756063082a0d56"),
    word: "capable",
    sentence: [
        "You can capable of better work than this.",
        "She's a very capable teacher."
    ],
    similar: [ ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e01dfe5f5756063082a0d57"),
    word: "pale",
    sentence: [
        "pale with fear",
        "You look pale, Are you ok"
    ],
    similar: [ ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e01dfe5f5756063082a0d58"),
    word: "tendency",
    sentence: [
        "I have a tendency to talk too much when I'm nervous.",
        "to display artistic, etc. tendency"
    ],
    similar: [ ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e01dfe5f5756063082a0d59"),
    word: "regard",
    sentence: [
        "He was driving without regard to speed limits."
    ],
    similar: [ ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e032022805a00e4b6af3c61")
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e0331d7528cb1c9fe215a82"),
    word: "rely",
    "sentence:": null,
    similar: [ ],
    synonymous: [ ],
    antonym: [ ],
    sentence: [
        "As babies, we rely entirely on others for food",
        "These days we rely heavily on computers to organize our work"
    ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e0331d7528cb1c9fe215a83"),
    word: "aware",
    "sentence:": null,
    similar: [ ],
    synonymous: [ ],
    antonym: [ ],
    sentence: [
        "As you're aware, this is not a new new problem."
    ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e0331d7528cb1c9fe215a84"),
    word: "wire",
    "sentence:": null,
    similar: [ ],
    synonymous: [ ],
    antonym: [ ],
    sentence: [
        "The box was fastened with a rusty wire."
    ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e0331d7528cb1c9fe215a85"),
    word: "accommodation",
    "sentence:": null,
    similar: [
        "command",
        "recommend"
    ],
    synonymous: [ ],
    antonym: [ ],
    sentence: [
        "rented/temporary accommodation"
    ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e0331d7528cb1c9fe215a86"),
    word: "benefit",
    sentence: [
        "We should spend the money on something that will benifit everyone."
    ],
    similar: [ ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e0331d7528cb1c9fe215a87"),
    word: "devote",
    sentence: [
        "She devoted herself to her career"
    ],
    similar: [ ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e0331d7528cb1c9fe215a88"),
    word: "praise",
    sentence: [
        "She praised his cooking.",
        "He praised his team for their performance."
    ],
    similar: [
        "parse"
    ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e0331d7528cb1c9fe215a89"),
    word: "track",
    sentence: [
        "a muddy track thought the forest",
        "tyre tracks",
        "railway tracks"
    ],
    similar: [ ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e04841972ada824609cd0d3"),
    word: "breast",
    sentence: [
        "breast cancer",
        "breast milk"
    ],
    similar: [
        "bread",
        "beard",
        "chest"
    ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e04841972ada824609cd0d4"),
    word: "shallow",
    sentence: [
        "These fish are found in shallow waters around the coast."
    ],
    similar: [ ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e04841972ada824609cd0d5"),
    word: "march",
    sentence: [
        "protest marchs",
        "to go on a march"
    ],
    similar: [
        "match"
    ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e04841972ada824609cd0d6"),
    word: "suffer",
    sentence: [
        "He suffers from asthma.",
        "I hate to see animals suffering"
    ],
    similar: [ ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e04841972ada824609cd0d7"),
    word: "trial",
    sentence: [
        "a murder trial",
        "He's on trial for murder."
    ],
    similar: [ ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e04841972ada824609cd0d8"),
    word: "remaining",
    sentence: [
        "The remaining twenty patients were transferred to another hospitol."
    ],
    similar: [ ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e04841972ada824609cd0d9"),
    word: "ideal",
    sentence: [
        "political ideal",
        "It's my ideal of what a family home shoud be."
    ],
    similar: [
        "idea",
        "deal"
    ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e04841972ada824609cd0da"),
    word: "aspect",
    sentence: [
        "The book aims to cover all aspects of city life."
    ],
    similar: [
        "respect"
    ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e04841972ada824609cd0db"),
    word: "manufacture",
    sentence: [
        "a car/computer manufacture",
        "Always follow the manufacture's instruction"
    ],
    similar: [ ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e04841972ada824609cd0dc"),
    word: "urban",
    sentence: [
        "urban areas",
        "urban life"
    ],
    similar: [ ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e09ff895ad7d7aedcf22390"),
    word: "thickness",
    sentence: [
        "Use wood of at least 12mm thickness"
    ],
    similar: [ ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e09ff895ad7d7aedcf22391"),
    word: "prevent",
    sentence: [
        "The accident could have been prevented"
    ],
    similar: [ ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e09ff895ad7d7aedcf22392"),
    word: "district",
    sentence: [
        "the city of London's financial district",
        "a school district"
    ],
    similar: [
        "area"
    ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e09ff895ad7d7aedcf22393"),
    word: "criticism",
    sentence: [
        "Th Plan has attracted criticism from consumer groups.",
        "I didn't mean it as a criticism."
    ],
    similar: [
        "blamed"
    ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e09ff895ad7d7aedcf22394"),
    word: "immroal",
    sentence: [
        "It's immoral to steal."
    ],
    similar: [
        "immoral"
    ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e09ff895ad7d7aedcf22395"),
    word: "removal",
    sentence: [
        "stain removal",
        "the removal of a tumour"
    ],
    similar: [
        "remove"
    ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e09ff895ad7d7aedcf22396"),
    word: "crisis",
    sentence: [
        "a political/financial crisis"
    ],
    similar: [ ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e09ff895ad7d7aedcf22397"),
    word: "ruined",
    sentence: [
        "a ruined castle"
    ],
    similar: [ ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e09ff895ad7d7aedcf22398"),
    word: "apparent",
    sentence: [
        "their devotion was apparent"
    ],
    similar: [ ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e09ff895ad7d7aedcf22399"),
    word: "flavour",
    sentence: [
        "The tomatoes give extra flavour to the sauce."
    ],
    similar: [ ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e09ff895ad7d7aedcf2239a"),
    word: "vocabulary",
    sentence: [
        "to have a wide/limited vocabulary"
    ],
    similar: [ ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e0b3c62c41c68a58286c2c2"),
    word: "sincere",
    sentence: [
        "a sincere attempt to resolve the problem",
        "sincere regret"
    ],
    similar: [ ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e0b3c62c41c68a58286c2c3"),
    word: "conventional",
    sentence: [
        "conventional behavior",
        "She's very conventional in her views",
        "conventional medicine"
    ],
    similar: [ ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e0b3c62c41c68a58286c2c4"),
    word: "apparently",
    sentence: [
        "Apparently they are getting divorced soon."
    ],
    similar: [ ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e0b3c62c41c68a58286c2c5"),
    word: "strict",
    sentence: [
        "strict rules/regulations",
        "She's on a very strict diet"
    ],
    similar: [
        "strike",
        "Burrowstrike"
    ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e0b3c62c41c68a58286c2c6"),
    word: "prior",
    sentence: [
        "Visite are by prior arrangement"
    ],
    similar: [ ],
    synonymous: [ ],
    antonym: [ ]
} ]);
db.getCollection("words").insert([ {
    _id: ObjectId("5e0b3c62c41c68a58286c2c7"),
    word: "guard",
    sentence: [
        "a security guard",
        "the captain of the guard"
    ],
    similar: [ ],
    synonymous: [ ],
    antonym: [ ]
} ]);
